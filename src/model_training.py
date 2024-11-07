import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize, LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle
from sentence_transformers import SentenceTransformer
from utils import create_directories, load_data_with_error_handling, save_model_with_error_handling, save_dataframe_to_excel

# Set output directories
create_directories(["output/model", "output/embeddings", "output/reports"])

# Load data with error handling
data = load_data_with_error_handling("path/to/your/dataset.xlsx")
data['Claim Description'] = data['Claim Description'].astype(str)

# Filter out extremely rare classes (those with fewer than 2 instances)
min_class_count = 2
X = data['Claim Description']
y_coverage = data['Coverage Code']
y_accident = data['Accident Source']

class_counts_coverage = y_coverage.value_counts()
classes_to_keep_coverage = class_counts_coverage[class_counts_coverage >= min_class_count].index

class_counts_accident = y_accident.value_counts()
classes_to_keep_accident = class_counts_accident[class_counts_accident >= min_class_count].index

X_filtered = X[y_coverage.isin(classes_to_keep_coverage) & y_accident.isin(classes_to_keep_accident)]
y_coverage_filtered = y_coverage[y_coverage.isin(classes_to_keep_coverage)]
y_accident_filtered = y_accident[y_accident.isin(classes_to_keep_accident)]

# Convert Claim Descriptions to embeddings using SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = np.vstack(X_filtered.progress_apply(lambda x: model.encode(x)).values)

# Encode target variables (Coverage Code and Accident Source)
coverage_encoder = LabelEncoder()
accident_encoder = LabelEncoder()

y_coverage_encoded = coverage_encoder.fit_transform(y_coverage_filtered)
y_accident_encoded = accident_encoder.fit_transform(y_accident_filtered)

# Split data into train/test
X_train, X_test, y_train_coverage, y_test_coverage, y_train_accident, y_test_accident = train_test_split(
    X_embeddings, y_coverage_encoded, y_accident_encoded, test_size=0.2, random_state=42, stratify=[y_coverage_encoded, y_accident_encoded]
)

# Apply SMOTE oversampling to address class imbalance for both targets
smote = SMOTE(random_state=42)
X_train_resampled, y_train_coverage_resampled, y_train_accident_resampled = smote.fit_resample(X_train, y_train_coverage, y_train_accident)

# Calculate class weights
class_weights_coverage = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_coverage_resampled), y=y_train_coverage_resampled)
class_weight_dict_coverage = dict(zip(np.unique(y_train_coverage_resampled), class_weights_coverage))

class_weights_accident = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_accident_resampled), y=y_train_accident_resampled)
class_weight_dict_accident = dict(zip(np.unique(y_train_accident_resampled), class_weights_accident))

# Instantiate RandomForestClassifier for both targets
rf_model_coverage = RandomForestClassifier(random_state=42, class_weight=class_weight_dict_coverage)
rf_model_accident = RandomForestClassifier(random_state=42, class_weight=class_weight_dict_accident)

# Train the models
rf_model_coverage.fit(X_train_resampled, y_train_coverage_resampled)
rf_model_accident.fit(X_train_resampled, y_train_accident_resampled)

# Evaluate the models
y_pred_coverage = rf_model_coverage.predict(X_test)
y_pred_accident = rf_model_accident.predict(X_test)

# Classification report for both targets
report_coverage = classification_report(y_test_coverage, y_pred_coverage, target_names=coverage_encoder.classes_)
report_accident = classification_report(y_test_accident, y_pred_accident, target_names=accident_encoder.classes_)

# Precision and Recall for both targets
precision_coverage = precision_score(y_test_coverage, y_pred_coverage, average="weighted", zero_division=1)
recall_coverage = recall_score(y_test_coverage, y_pred_coverage, average="weighted", zero_division=1)

precision_accident = precision_score(y_test_accident, y_pred_accident, average="weighted", zero_division=1)
recall_accident = recall_score(y_test_accident, y_pred_accident, average="weighted", zero_division=1)

# Multi-class ROC AUC for both targets
y_test_coverage_binarized = label_binarize(y_test_coverage, classes=np.unique(y_coverage_filtered))
y_test_accident_binarized = label_binarize(y_test_accident, classes=np.unique(y_accident_filtered))

roc_auc_coverage = roc_auc_score(y_test_coverage_binarized, rf_model_coverage.predict_proba(X_test), average="weighted", multi_class="ovr")
roc_auc_accident = roc_auc_score(y_test_accident_binarized, rf_model_accident.predict_proba(X_test), average="weighted", multi_class="ovr")

# Save the models
save_model_with_error_handling(rf_model_coverage, "output/model/optimized_rf_model_coverage.pkl")
save_model_with_error_handling(rf_model_accident, "output/model/optimized_rf_model_accident.pkl")

# Save evaluation results to Excel
eval_results = pd.DataFrame({
    "Target": ["Coverage Code", "Accident Source"],
    "Precision": [precision_coverage, precision_accident],
    "Recall": [recall_coverage, recall_accident],
    "ROC AUC": [roc_auc_coverage, roc_auc_accident]
})

save_dataframe_to_excel({"Evaluation Results": eval_results}, "output/reports/evaluation_results.xlsx")

# Print the reports
print("Coverage Code Report:\n", report_coverage)
print("Accident Source Report:\n", report_accident)
