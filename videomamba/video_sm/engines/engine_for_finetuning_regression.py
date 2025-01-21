import os
import time
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
import torch.distributed as dist
from timm.utils import ModelEma
import utils
import itertools

torch.autograd.set_detect_anomaly(True)

def log_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            max_grad = torch.max(parameter.grad)
            min_grad = torch.min(parameter.grad)
            if torch.isnan(max_grad) or torch.isnan(min_grad):
                print(f"NaN gradient at {name}")

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)[:, 0]
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    try:
        return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale
    except Exception:
        return 0


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, args, max_norm: float = 0,
                    resume_step=0, save_freq=1000, model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, no_amp=False, bf16=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    data_iter = itertools.islice(enumerate(metric_logger.log_every(data_loader, print_freq, header)), resume_step, None)

    train_nan_counter = 0
    loss_nan_counter = 0

    for data_iter_step, (samples, targets, _, _) in data_iter:
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue

        total_previous_steps = args.resume_step + (epoch * num_training_steps_per_epoch) # Need to fix this because we only use resume steps at the start (?)
        it = total_previous_steps + data_iter_step  # global training iteration + prev steps
        # data_iter_step is basically the resume_step (initially), which is taken into account in total_previous_steps
        # it = total_previous_steps 
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    if "lr_scale" in param_group:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"] * args.scale_lr
                    else:
                        param_group["lr"] = lr_schedule_values[it] * args.scale_lr
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).to(torch.float32)

        if torch.isnan(samples).any() or torch.isnan(targets).any():
            train_nan_counter += 1
            print(f"VideoMamba (train): {train_nan_counter} NaN detected in samples or targets. Skipping.")
            continue

        if loss_scaler is None:
            if not no_amp and not args.high_precision:
                samples = samples.bfloat16() if bf16 else samples.half()
            loss, output = train_class_batch(model, samples, targets, criterion)
            if args.log_mae:
                output = output.detach()
                with torch.no_grad():
                    mae_loss = torch.nn.L1Loss()(output, targets)
        else:
            with amp_autocast:
                loss, output = train_class_batch(model, samples, targets, criterion)
                if args.log_mae:
                    output = output.detach()
                    with torch.no_grad():
                        mae_loss = torch.nn.L1Loss()(output, targets)

        loss_value = loss.item()
        if args.log_mae:
            mae_value = mae_loss.item()

        # loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        # dist.all_gather(loss_list, loss)
        # loss_list = torch.tensor(loss_list)
        # loss_list_isnan = torch.isnan(loss_list).any()
        # loss_list_isinf = torch.isinf(loss_list).any()

        # if loss_list_isnan or loss_list_isinf:
        #     print(" ========== loss_isnan = {},  loss_isinf = {} ========== ".format(loss_list_isnan, loss_list_isinf))
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        if not math.isfinite(loss_value):
            if args.skip_nan:
                loss_nan_counter += 1
                print(f"Loss is {loss_value}, skipping batch ({loss_nan_counter} skipped)")
                continue
            else:
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
                

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            if loss_scaler != 'none':
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss /= update_freq
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order,
                                        update_grad=(data_iter_step + 1) % update_freq == 0)
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                loss_scale_value = loss_scaler.state_dict()["scale"]
            else:
                loss /= update_freq
                loss.backward()
                if (data_iter_step + 1) % update_freq == 0:
                    if max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    log_gradients(model)
                    optimizer.step()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                loss_scale_value = 0

        torch.cuda.synchronize()

        if (data_iter_step + 1) % save_freq == 0:
            model_without_ddp = model
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, step=data_iter_step, epoch=epoch, model_name='latest', model_ema=model_ema)
            print(f"Saved model to checkpoint-{data_iter_step}.pth at step {data_iter_step}.")

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if args.log_mae:
            metric_logger.update(mae=mae_value)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            if args.log_mae:
                log_writer.update(mae=mae_value, head="loss")

            log_writer.set_step(it)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, amp_autocast, args, ds=True, no_amp=False, bf16=False):
    criterion = torch.nn.MSELoss()

    if args.log_mae:
        mae_criterion = torch.nn.L1Loss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()
    val_nan_counter = 0

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).to(torch.float32)

        if torch.isnan(videos).any() or torch.isnan(target).any():
            val_nan_counter += 1
            print("VideoMamba (val): {val_nan_counter} NaN detected in videos or target. Skipping.")
            continue

        # compute output
        if ds:
            if not no_amp and not args.high_precision:
                videos = videos.bfloat16() if bf16 else videos.half()
            output = model(videos)[:, 0]
            loss = criterion(output, target)
            if args.log_mae:
                mae_loss = mae_criterion(output, target)
        else:
            with amp_autocast:
                output = model(videos)[:, 0]
                loss = criterion(output, target)
                if args.log_mae:
                    mae_loss = mae_criterion(output, target)

        metric_logger.update(loss=loss.item())
        if args.log_mae:
            metric_logger.update(mae=mae_loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    if args.log_mae:
        print('* mae {mae_values.global_avg:.3f}'.format(mae_values=metric_logger.mae))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file, amp_autocast, ds=True, no_amp=False, bf16=False):
    criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).to(torch.float32)

        # compute output
        if ds:
            if not no_amp and not args.high_precision:
                videos = videos.bfloat16() if bf16 else videos.half()
            output = model(videos)[:, 0]
            loss = criterion(output, target)
        else:
            with amp_autocast:
                output = model(videos)[:, 0]
                loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].float().cpu().numpy()), \
                                                str(float(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        metric_logger.update(loss=loss.item())

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}\n".format(loss.item()))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('*loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip().split(' ')
            name = line[0]
            data = float(line[1])
            label = float(line[2])
            chunk_nb = line[3]
            split_nb = line[4]
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    mse_loss = [x[0] for x in ans]
    final_loss = np.mean(mse_loss)
    return final_loss

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    mse_loss = (feat - label) ** 2.0
    return [mse_loss]
