from sklearn.model_selection import train_test_split
import random

"""
python3 step1_0_group_balanced_train_test_split.py  
"""

# : change data folder if necessary

data_root = '/home/linuxuser/user/data/pazienti/'

# Known indices for each group are stored in dictionary
indices = {
    'Psy':  [3, 5, 6, 19, 24, 25, 45, 52, 56, 63, 67, 71, 76, 81, 90, 91, 100, 101, 102],  
    'MCI':  [11, 15, 18, 20, 28, 29, 33, 34, 38, 42, 47, 49, 53, 61, 72, 73, 83, 88, 97, 99],
    'AD': [4, 7, 8, 10, 12, 13, 17, 26, 27, 30, 35, 36, 37, 39, 40, 41, 43, 48, 50, 51, 54, 55, 57, 58, 59, 62, 64, 66, 68, 69, 70, 75, 77, 78, 82, 84, 85, 87, 94, 95, 96, 98, 104],
    'other': [1, 2, 9, 14, 16, 21, 22, 23, 31, 32, 44, 46, 60, 65, 74, 79, 80, 86, 89, 92, 93, 103]
}

train_indices = []
test_indices = []

# Split ratio for training (in our case., 75% training, 25% testing)
train_ratio = 0.75

# Use seed for reproducibility
random.seed(42)

for group_indices in indices.values():
    # Shuffle the indices within each group
    random.shuffle(group_indices)
    
    # Split into train and test
    train, test = train_test_split(group_indices, test_size=(1 - train_ratio), random_state=42)
    
    train_indices.extend(train)
    test_indices.extend(test)

# Optionally shuffle the combined sets
#random.shuffle(train_indices)
#random.shuffle(test_indices)

# Check that no indices are duplicated or lost
assert len(set(train_indices)) == len(train_indices)
assert len(set(test_indices)) == len(test_indices)
assert len(set(train_indices) & set(test_indices)) == 0
assert set(train_indices) | set(test_indices) == set(range(1, 105))

# Write the indices to one .txt file with two columns "train_ids", "test_ids" in folder "/home/linuxuser/user/data/pazienti". 
# Assure that all numbers are saved using 3 digits, e.g., 001, 002, 003, ..., 104.
# The columns do not have the same length. 
with open(data_root + 'train_test_split.txt', 'w') as f:
    f.write("train_ids,")
    for train_id in train_indices:
        f.write(f"{train_id:03},")
    f.write("\n")
    f.write("test_ids,")
    for test_id in test_indices:
        f.write(f"{test_id:03},")

print("indices written to path: ", data_root + 'train_test_split.txt')


 

    








