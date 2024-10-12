import pandas as pd
import matplotlib.pyplot as plt


# Load the Excel file
file_path = '/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/train_test_results.xlsx'

# Load the sheets based on the correct sheet names
train_results = pd.read_excel(file_path, sheet_name='Train')
test_results = pd.read_excel(file_path, sheet_name='Test')

# Display the first few rows of each sheet to understand the structure
train_results.head(), test_results.head()



# Adjusting the column names for the training results dataframe
train_dice_scores = train_results[['Ex. Train', 'Toolbox ', '(Best) Dice']].dropna()

# Adjusting the column names for the testing results dataframe
test_dice_scores = test_results[['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 3']].dropna()
test_dice_scores.columns = ['Experiment', 'Toolbox', 'Mean Dice Score']

# Plotting the training Dice scores
plt.figure(figsize=(10, 6))
for toolbox in train_dice_scores['Toolbox '].unique():
    toolbox_data = train_dice_scores[train_dice_scores['Toolbox '] == toolbox]
    plt.plot(toolbox_data['Ex. Train'], toolbox_data['(Best) Dice'], label=f'{toolbox} - Train')

plt.title('Training Dice Scores Across Experiments and Toolboxes')
plt.xlabel('Experiment')
plt.ylabel('Best Dice Score')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the testing Dice scores for different ground truths
plt.figure(figsize=(10, 6))
for toolbox in test_dice_scores['Toolbox'].unique():
    toolbox_data = test_dice_scores[test_dice_scores['Toolbox'] == toolbox]
    plt.plot(toolbox_data['Experiment'], toolbox_data['Mean Dice Score'], label=f'{toolbox} - Test')

plt.title('Testing Dice Scores Across Experiments and Toolboxes')
plt.xlabel('Experiment')
plt.ylabel('Mean Dice Score')
plt.legend()
plt.grid(True)
plt.show()
