import clip.clip as clip
import os
import torch
from collections import OrderedDict

path = '/cluster/home/t115318uhn/VideoMamba/videomamba/video_sm/models'

model, _ = clip.load("ViT-B/16", device='cpu')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        new_state_dict[k[7:]] = v
torch.save(new_state_dict, os.path.join(path, 'vit_b16.pth'))


model, _ = clip.load("ViT-L/14", device='cpu')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        new_state_dict[k[7:]] = v
torch.save(new_state_dict, os.path.join(path, 'vit_l14.pth'))


model, _ = clip.load("ViT-L/14@336px", device='cpu')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        new_state_dict[k[7:]] = v
torch.save(new_state_dict, os.path.join(path, 'vit_l14_336.pth'))