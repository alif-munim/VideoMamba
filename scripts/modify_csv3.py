import pandas as pd

# Load the CSV file
df = pd.read_csv('modified_video_data2.csv', header=None)

# Define the part to be removed
part_to_remove = '/scratch/alif/VideoMamba/echo_data/'

# Remove the specified part from each line in the first column
df[0] = df[0].str.replace(part_to_remove, '')

# Save the modified DataFrame back to a new CSV file
df.to_csv('cleaned_video_data.csv', index=False, header=False)
