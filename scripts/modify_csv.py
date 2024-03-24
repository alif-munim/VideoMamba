import pandas as pd

# Define the original and new file paths
original_csv_path = 'video_data.csv'
modified_csv_path = 'modified_video_data.csv'

# Define the path to prepend
prepend_path = '/scratch/alif/VideoMamba/'

# Read the original CSV file
df = pd.read_csv(original_csv_path)

# Rename the columns
df.rename(columns={'Video Path': 'path', 'Length': 'length', 'Label': 'label'}, inplace=True)

# Prepend the specified path to each video path
df['path'] = prepend_path + df['path'].astype(str)

# Write the modified data to a new CSV file
df.to_csv(modified_csv_path, index=False)

print(f'Modified CSV has been saved to {modified_csv_path}')
