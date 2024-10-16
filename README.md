# soz_classifier_ml
Seizure Onset Code for Classification and Analysis

Overview

Hello this is an accumulation of work I produced which involved creating/editing a classifier to aid in soz research below is a brief understanding of the example, since the summarizer and graph are pretty self explanatory, I will explain 
quickly. The summarizer essentially grabbed the CSV files that was created from the various classification testing and summarized into a csv which had differing sparsity results, the makegraph leveraged libraries to turn the results into graph plots. Below is more indepth work and specifics related to the seizure onset machine learning work.

This project focuses on detecting the Seizure Onset Zone (SOZ), which is a critical region of the brain responsible for triggering seizures in individuals with epilepsy. Accurate identification of the SOZ is important for guiding surgical interventions and improving patient outcomes. We aim to classify whether a brain region belongs to the SOZ based on neurophysiological data using machine learning models.

In general, to create greater understanding we utilized machine learning classifiers to predict whether a given sample of brain activity belongs to the SOZ or non-SOZ areas. The classifiers are trained using various features derived from neurophysiological recordings. The code includes cross-validation, feature ranking, and parameter tuning to optimize model performance.

Data
The input data for this project consists of neurophysiological features recorded from different brain regions during the preictal (before seizure) and ictal (during seizure) phases. The features include:

BC, CF, HF, etc.: These are signal features extracted from the brain regions.
SOZ_label: The class label for each sample, where 1 indicates the region is part of or close to the SOZ, and 0 indicates it is not.
SID: Subject identifier, used to ensure that the model does not leak information across subjects during cross-validation.
Objective
The primary goal is to build machine learning models that can classify brain regions into SOZ and non-SOZ classes. This project involves:

Cross-Validation: Evaluating the model’s performance using GroupKFold, ensuring subject-level splitting for generalization.
Feature Ranking: Identifying and ranking the most important features that contribute to accurate classification using Mutual Information (MI) and F-statistic.
Class Imbalance Handling: The class distribution is imbalanced (fewer SOZ regions), so we use SMOTE to oversample the minority class during training.
Classifier Models: Several classifiers are used, including SVM (Support Vector Machine), MLP (Multilayer Perceptron), and Extra Trees Classifier, among others. Hyperparameters are optimized using a grid search.
Workflow
1. Feature Scaling and Selection
The input features are scaled using StandardScaler, which is critical for algorithms like SVM and neural networks. We also rank the features using two methods:

Mutual Information (MI) measures how much each feature contributes to predicting the target class.
F-statistic is used to assess the feature importance in a statistical sense.
The combination of these metrics allows us to order the features by importance for each cross-validation fold.

2. SMOTE Oversampling
Since the data is imbalanced, we use SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic samples of the minority class (SOZ regions), allowing the classifiers to learn a more balanced representation of the two classes.

3. Cross-Validation
We use GroupKFold cross-validation to ensure that samples from the same subject are not split between training and test sets. This approach helps to avoid overfitting and ensures that the model generalizes across different subjects.

4. Model Training and Evaluation
We train the models on different subsets of features, ranging from the top-ranked single feature to the full set of features. The models are evaluated based on:

Accuracy: Measures the overall classification performance.
ROC AUC: Measures the area under the receiver operating characteristic curve, providing insight into the model’s ability to distinguish between classes.
Sensitivity: The model’s ability to correctly identify SOZ regions.
Specificity: The model’s ability to correctly classify non-SOZ regions.
Confusion Matrix: Provides a detailed breakdown of the classification results, including true positives, false positives, true negatives, and false negatives.
5. Results
The models' performance is logged across multiple iterations and folds, with detailed records of feature importance, accuracy, ROC AUC, sensitivity, and specificity. These results are saved as CSV files for further analysis.
