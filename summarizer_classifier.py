import pandas as pd
import glob
import os
import ast

def parse_hyperparameters(hyperparam_str):
    #converts hyperparameters to string represenations
    try:
        hyperparams = ast.literal_eval(hyperparam_str)
        return ', '.join(f"{k}={v}" for k, v in hyperparams.items())
    except:
        return hyperparam_str

def process_files(input_dir, output_dir, classifier_prefix):
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"Failed to create directory {output_dir}: {e}")
            return

    metrics = ['Train_Accuracy', 'Train_AUC', 'Train_Sensitivity', 'Train_Specificity', 'Test_Accuracy', 'Test_AUC', 'Test_Sensitivity', 'Test_Specificity']
    #look for classifier files (CSV) in hte input directory
    files = glob.glob(os.path.join(input_dir, f'{classifier_prefix}*.csv'))
    print(f"Files found: {files}")  # Debugging: Check if files are being found

    if not files:
        print("No files found matching the pattern.")
        return
    #process the files
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            df['Hyperparameters'] = df['Hyperparameters'].apply(parse_hyperparameters)
            summary_list = []
            #group data by hyperparameters and iterate over groups
            for (hyperparams), group in df.groupby('Hyperparameters'):
                #process feature count number 1 through 10
                for feat_count in range(1, 11):
                    feature_group = group[group['N_feats'] == feat_count]
                    if not feature_group.empty:
                        summary_data = {
                            'Hyperparameters': hyperparams,
                            'Feature Count': feat_count
                        }
                        #calculate mean and standard deviation
                        for metric in metrics:
                                mean_value = feature_group[metric].mean()
                                sd_value = feature_group[metric].std()
                                summary_data[f'{metric}'] = f"{mean_value} + {sd_value}"
                        summary_list.append(summary_data)
            
            summary_df = pd.DataFrame(summary_list)
            output_file = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(file_path))[0]}_summary.csv')
            summary_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")  #to debug
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

# Example usage
input_directory = r'C:\Users\Melon\Desktop\Research\Merged_CSV_Classifier\preictal_ict20'
output_directory = r'C:\Users\Melon\Desktop\Research\classifier_summary\preictal_ict20'
classifier_prefix = 'ETC_'  # Specify the classifier identifier
process_files(input_directory, output_directory, classifier_prefix)
