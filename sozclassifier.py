import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from itertools import product
from sklearn.feature_selection import f_classif,mutual_info_classif
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fnum',type=int, required=True)
parser.add_argument('--cv_k',type=int, required=True)
args = parser.parse_args()


# Define a function for cross-validation
def cross_validate_model(X, y, cols, groups, model_name, model, param_grid, cv_k=15, sc=1, oversample = 0, n_iter = 1):
    """
    X           - Input array features n_cols = n_features (10 here)
    y           - Class label (0 for non-SOZ and 1 for SOZ and its immediate neighbors)
    cols        - List with the column names
    groups      - One hot encoded group ID for group-level splitting
    model_name  - Informal name to describe the classifier model
    model       - Class name for the model
    param_grid  - Dictionary containing the model parameters
    cv_k        - Number of folds for GroupKFold
    sc          - Flag for scaling, use standard scaling if sc == 1
    n_iter      - Number of times to repeat the entire process from splitting the data into K folds to train and test the models
    """
    
    if model == MLPClassifier:
        sc, oversample = 1,1
        
    n_feats = X.shape[1]
    model_results = []
    for it in range(1,n_iter+1):
      cv = GroupKFold(n_splits=cv_k) # Splitting into k groups for each iteration
      print("----------------------------------------------------------")
      print("Iteration "+str(it))
      fold = 0
      for train_indices, test_indices in cv.split(X, y, groups=groups):

          # For each fold
          fold = fold + 1
          print("Fold "+str(fold))
          X_train_original, X_test_original = X[train_indices], X[test_indices]
          y_train, y_test = y[train_indices], y[test_indices]

          # Scale the data - can be done simultaneously since it scales the features independently
          if sc!=0:
              scaler = StandardScaler()
              X_train_original = scaler.fit_transform(X_train_original)
              X_test_original = scaler.transform(X_test_original)

          # Perform feature ranking using MI and F statistic
          f_statistic, _ = f_classif(X_train_original, y_train)
          mi = mutual_info_classif(X_train_original,y_train)

          f_ind = np.argsort(f_statistic)
          mi_ind = np.argsort(mi)

          overall_score = np.zeros(n_feats)

          for i in range(n_feats):
              overall_score[i] += np.flatnonzero(f_ind==i).size + np.flatnonzero(mi_ind==i).size

          scored_feat_order = np.flip(np.argsort(overall_score))
          sorted_cols = [cols[index] for index in scored_feat_order] # List array sorted by feature ranks for the fold          

          # Choose top k features at a time
          for top_k in range(1,n_feats+1): # Adding 1 since python index starts with 0
            print("n_feats "+str(top_k))
            X_train = X_train_original[:,scored_feat_order[:top_k]] 
            X_test  = X_test_original[:,scored_feat_order[:top_k]] 
          
            # SMOTE Over-sampling
            if oversample == 1:
                smote = SMOTE(sampling_strategy='auto', random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            model_instance = model()
            for params in product(*param_grid.values()):
                print(params)
                param_dict = dict(zip(param_grid.keys(), params))
                model_instance.set_params(**param_dict)

                # Train the model on the training data
                model_instance.fit(X_train_resampled, y_train_resampled)

                # Predict on the training and test data
                y_train_pred = model_instance.predict_proba(X_train_resampled)
                y_test_pred = model_instance.predict_proba(X_test)

				#Calculate predictions, accuracy, and confusion matrix
                train_predictions = np.argmax(y_train_pred, axis=1)
                test_predictions = np.argmax(y_test_pred, axis=1)

				# Calculate accuracy
                train_accuracy = accuracy_score(y_train_resampled, train_predictions)
                test_accuracy = accuracy_score(y_test, test_predictions)
    
				#Calculate ROC_AUC
                num_classes = len(np.unique(y_train_resampled))
                if num_classes == 2:
                    train_auc = roc_auc_score(y_train_resampled, y_train_pred[:,1])
                    test_auc = roc_auc_score(y_test, y_test_pred[:,1])
                else:
                    train_auc = roc_auc_score(y_train_resampled, y_train_pred, multi_class='ovo')
                    test_auc = roc_auc_score(y_test, y_test_pred, multi_class='ovo')
					
				# Calculate confusion matrix
                cm_train = confusion_matrix(y_train_resampled, train_predictions)
                cm_test = confusion_matrix(y_test, test_predictions)
				
				#Calculate specificity to avoid runtime warning
                sum_predictions_train = np.sum(cm_train, axis=0)
                true_positives_train = np.diag(cm_train)
                train_specificity = np.divide(true_positives_train, sum_predictions_train, where=(sum_predictions_train != 0))
                train_specificity = np.nanmean(train_specificity)  # Calculate the mean, ignoring NaNs

                sum_predictions_test = np.sum(cm_test, axis=0)
                true_positives_test = np.diag(cm_test)
                test_specificity = np.divide(true_positives_test, sum_predictions_test, where=(sum_predictions_test != 0))
                test_specificity = np.nanmean(test_specificity)  # Calculate the mean, ignoring NaNs

				# Handle both binary and multi-class cases
                if cm_train.shape == (2, 2):  # Binary classification case
                    tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
                    train_sensitivity = tp_train / (tp_train + fn_train) if (tp_train + fn_train) != 0 else 0
                else:  # Multi-class classification case
                    sensitivities_train = np.diag(cm_train) / np.sum(cm_train, axis=1, where=np.sum(cm_train, axis=1) != 0)
                    train_sensitivity = np.nanmean(sensitivities_train)  # Mean of sensitivities for each class
				# Do similarly for the test data
                if cm_test.shape == (2, 2):
                    tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
                    test_sensitivity = tp_test / (tp_test + fn_test) if (tp_test + fn_test) != 0 else 0
                else:
                    sensitivities_test = np.diag(cm_test) / np.sum(cm_test, axis=1, where=np.sum(cm_test, axis=1) != 0)
                    test_sensitivity = np.nanmean(sensitivities_test)

                model_results.append({
                    'Iteration': it,
                    'Model': model_name,
                    'Fold':fold,
                    'N_feats':top_k,
                    'Feat_order':sorted_cols,
                    'Hyperparameters': param_dict,
                    'Train_Accuracy': train_accuracy,
                    'Train_AUC': train_auc,
                    'Train_Sensitivity': train_sensitivity,
                    'Train_Specificity': train_specificity,
                    'Test_Accuracy': test_accuracy,
                    'Test_AUC': test_auc,
                    'Test_Sensitivity': test_sensitivity,
                    'Test_Specificity': test_specificity
                })
    return model_results

data_dir = "/home/parhikk/ng000090/Research/Merged_CSV/ictal30/"
res_dir = "/home/parhikk/ng000090/Research/Merged_CSV_Classifier/preictal2min/"

fnum = args.fnum # File number - 0 - 59
cv_k = args.cv_k

# Load the data as a Pandas DataFrame
os.chdir(data_dir)
d = os.listdir()
filename = d[fnum]
data = pd.read_csv(filename)

# Define the features, labels, and group ID
y = data['SOZ_label']
groups = data['SID'].to_list()
X = data.loc[:, 'BC':'LE']
col_names = list(X.columns)

# Convert features, labels, and group_id to numpy arrays
X1 = X.to_numpy()
y1 = y.to_numpy()
y1 = 1*(y1>=0) #used this because we compared -1 to 1 or 0 not 1 vs 0.
groups = [np.where(np.array(list(dict.fromkeys(groups)))==e)[0][0]for e in groups]

# Define a list of classifiers with their associated hyperparameter grids
models = [
    
    ('SVM (RBF)', SVC, {
		'probability': [True],
        'class_weight':['balanced'],
        'C': [0.1, 1,5,10],
        'gamma': ['scale', 0.01, 1, 2, 10]})
]    

# Save results
all_results = []
os.chdir(res_dir)
res_file = "SVM_"+str(cv_k) + "_fold_" + filename

parallel_results = Parallel(n_jobs=-1)(
    delayed(cross_validate_model)(X1, y1, col_names ,groups, model_name, model, param_dict,cv_k)
    for model_name, model, param_dict in models
    )
all_results = [result for results in parallel_results for result in results]
results_df = pd.DataFrame(all_results)
results_df.to_csv(res_file, index=False)
