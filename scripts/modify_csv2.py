import pandas as pd

# Load the CSV file, assuming no header as you want to remove it
df = pd.read_csv('modified_video_data.csv', header=0)  # Change 'header=0' if your data does have headers

# Remove the 'Length' column
df.drop('length', axis=1, inplace=True)

# Replace all values in the second column (now 'Label') with 1
df.iloc[:, 1] = 1

# Save back to CSV without the header and the 'Length' column
df.to_csv('modified_video_data2.csv', index=False, header=False)
