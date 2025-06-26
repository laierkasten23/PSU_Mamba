import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the provided CSV files to inspect their structure
file_t1_vs_ce = '/home/linuxuser/user/project_dir/_experiments/00_preanalysis/results_t1ce_vs_t1.csv'
file_t1xflair_vs_ce = '/home/linuxuser/user/project_dir/_experiments/00_preanalysis/results_t1ce_vs_t1xflair.csv'

# Reading the CSV files
df_t1_vs_ce = pd.read_csv(file_t1_vs_ce)
df_t1xflair_vs_ce = pd.read_csv(file_t1xflair_vs_ce)

# Display the first few rows to understand the structure of the data
df_t1_vs_ce.head(), df_t1xflair_vs_ce.head()

# Regenerating the boxplots to show side-by-side comparisons for T1 vs. CE-T1 and T1xFLAIR vs. CE-T1

# Creating a new DataFrame for easier plotting
df_combined = pd.DataFrame({
    'DSC_T1': df_t1_vs_ce['DSC'],
    'DSC_T1xFLAIR': df_t1xflair_vs_ce['DSC'],
    'Jaccard_T1': df_t1_vs_ce['Jaccard'],
    'Jaccard_T1xFLAIR': df_t1xflair_vs_ce['Jaccard'],
    'Kappa_T1': df_t1_vs_ce['Kappa'],
    'Kappa_T1xFLAIR': df_t1xflair_vs_ce['Kappa'],
    'Hausdorff_T1': df_t1_vs_ce['Hausdorff'],
    'Hausdorff_T1xFLAIR': df_t1xflair_vs_ce['Hausdorff'],
})

# Setting up the figure for boxplots
fig, axes = plt.subplots(1, 4, figsize=(20, 10))
fig.suptitle('Comparison of Metrics: T1w vs. CE-T1w and T1xFLAIR vs. CE-T1w', fontsize=26)

# Define a colorblind-friendly palette
palette = sns.color_palette("colorblind")

# Plotting side-by-side boxplots
sns.boxplot(data=df_combined[['DSC_T1', 'DSC_T1xFLAIR']], ax=axes[0], palette=palette)
axes[0].set_title('Dice Similarity Coefficient (DSC)', fontsize=22)
axes[0].set_xticklabels(['T1w vs\nCE-T1w', 'T1xFLAIR vs\nCE-T1w'], fontsize=18)
axes[0].set_ylabel('DSC', fontsize=20)
axes[0].tick_params(axis='both', which='major', labelsize=20)

sns.boxplot(data=df_combined[['Jaccard_T1', 'Jaccard_T1xFLAIR']], ax=axes[1], palette=palette)
axes[1].set_title('Jaccard Index', fontsize=22)
axes[1].set_xticklabels(['T1w vs\nCE-T1w', 'T1xFLAIR vs\nCE-T1w'], fontsize=18)
axes[1].set_ylabel('Jaccard Index', fontsize=20)
axes[1].tick_params(axis='both', which='major', labelsize=20)

sns.boxplot(data=df_combined[['Kappa_T1', 'Kappa_T1xFLAIR']], ax=axes[2], palette=palette)
axes[2].set_title('Cohen\'s Kappa', fontsize=22)
axes[2].set_xticklabels(['T1w vs\nCE-T1w', 'T1xFLAIR vs\nCE-T1w'], fontsize=18)
axes[2].set_ylabel('Kappa', fontsize=20)
axes[2].tick_params(axis='both', which='major', labelsize=20)

sns.boxplot(data=df_combined[['Hausdorff_T1', 'Hausdorff_T1xFLAIR']], ax=axes[3], palette=palette)
axes[3].set_title('Hausdorff Distance', fontsize=22)
axes[3].set_xticklabels(['T1w vs\nCE-T1w', 'T1xFLAIR vs\nCE-T1w'], fontsize=18)
axes[3].set_ylabel('Hausdorff Distance', fontsize=20)
axes[3].tick_params(axis='both', which='major', labelsize=20)

# Adjust layout to create more space for the labels
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()