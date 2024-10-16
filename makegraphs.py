import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

def plot_metric_boxplots(input_dir, output_dir, classifier_prefix):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define metrics to plot
    metrics = ['Train_Accuracy', 'Train_AUC', 'Train_Sensitivity', 'Train_Specificity', 
               'Test_Accuracy', 'Test_AUC', 'Test_Sensitivity', 'Test_Specificity']

    # Find all summary files for the given classifier prefix
    file_pattern = os.path.join(input_dir, f'{classifier_prefix}*_summary.csv')
    files = glob.glob(file_pattern)

    # Loop through each file
    for file in files:
        df = pd.read_csv(file)
        # Extract hyperparameters to label plots and manage output
        hyperparam_sets = df['Hyperparameters'].unique()

        # Process each hyperparameter set
        for hyperparams in hyperparam_sets:
            df_hyper = df[df['Hyperparameters'] == hyperparams]
            melted_df = df_hyper.melt(id_vars=['Feature Count'], value_vars=metrics,
                                      var_name='Metric', value_name='Value')

            # Plot each metric
            for metric in metrics:
                plt.figure(figsize=(40, 10))
                sns.boxplot(data=melted_df[melted_df['Metric'] == metric], x='Feature Count', y='Value', hue='Metric')
                plt.title(f'{metric} vs Feature Count\nHyperparameters: {hyperparams}')
                plt.xlabel('Number of Features')
                plt.ylabel(f'{metric} (Mean + SD)')
                plt.legend(title='Metric', loc='upper right')

                # Construct a filename based on metric and hyperparameters
                sanitized_hyperparams = hyperparams.replace(', ', '_').replace('=', '').replace(':', '').replace('|', '_')
                filename = f"{classifier_prefix}_{metric}_{sanitized_hyperparams}.png"
                plt.savefig(os.path.join(output_dir, filename))
                plt.close()

# Example usage
input_directory = r'C:\Users\Melon\Desktop\Research\classifier_summary\preictal_ict20'
output_directory = r'C:\Users\Melon\Desktop\Research\Graphs\preictal_ict20'
classifier_prefixes = ['RF_']  # List classifier prefixes if there are multiple
for prefix in classifier_prefixes:
    plot_metric_boxplots(input_directory, output_directory, prefix)
