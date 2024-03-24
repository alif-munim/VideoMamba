import cv2
import os
import pandas as pd
from tqdm import tqdm

# Define the source directory and the target directory for resized videos
source_dir = '/scratch/alif/echo-reports/data/echo_data'
target_dir = 'full_resized_echo_data'
os.makedirs(target_dir, exist_ok=True)  # Create target directory if it doesn't exist

# Prepare to collect video data
videos_data = []

# Collect all video paths first for progress bar
video_files = []
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.mp4'):
            video_files.append(os.path.join(root, file))

# Process video files
for video_path in tqdm(video_files, desc='Resizing Videos'):
    # Construct the path for the resized video
    rel_path = os.path.relpath(os.path.dirname(video_path), source_dir)
    target_subdir = os.path.join(target_dir, rel_path)
    os.makedirs(target_subdir, exist_ok=True)
    target_video_path = os.path.join(target_subdir, os.path.basename(video_path))
    
    # Open the source video
    cap = cv2.VideoCapture(video_path)
    # Obtain the video length (number of frames)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define the codec and create VideoWriter object to write the resized video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You may need to change this depending on your video codec
    out = cv2.VideoWriter(target_video_path, fourcc, 20.0, (224, 224))
    
    # Read and resize video frames, then write to the new video file
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (224, 224))
        out.write(resized_frame)
    
    # Release everything when job is finished
    cap.release()
    out.release()
    
    # Append video data for CSV
    videos_data.append([video_path, length, 1])

# Create a DataFrame and save to CSV
df = pd.DataFrame(videos_data, columns=['path', 'length', 'label'])
csv_file = 'video_data.csv'
df.to_csv(csv_file, index=False)
print(f'Video data has been saved to {csv_file}')
