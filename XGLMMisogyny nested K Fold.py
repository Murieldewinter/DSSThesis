import torch
from transformers import XGLMModel, XGLMTokenizer
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import msoffcrypto
import io

# Decrypt and read the training dataset
file_path_train = '/home/u161198/new_env/Datasets/es_AMI_TrainingSet_NEW (2).xlsx'
password_train = '!20AMI_ES18?'

file_train = msoffcrypto.OfficeFile(open(file_path_train, 'rb'))
file_train.load_key(password=password_train)

decrypted_stream_train = io.BytesIO()
file_train.decrypt(decrypted_stream_train)

decrypted_stream_train.seek(0)
Misogyny_train = pd.read_excel(decrypted_stream_train, engine='openpyxl')

print(Misogyny_train.head())

# Read the testing dataset
file_path_test = '/home/u161198/new_env/Datasets/es_AMI_TestSet.tsv'
Misogyny_test = pd.read_csv(file_path_test, delimiter='\t')

print(Misogyny_test.head())

# Extract texts and labels
texts_train = Misogyny_train['tweet'].tolist()
labels_train = Misogyny_train['misogynous'].values

texts_test = Misogyny_test['tweet'].tolist()
labels_test = Misogyny_test['misogynous'].values

# Load the model and tokenizer
model = XGLMModel.from_pretrained('facebook/xglm-7.5B')
tokenizer = XGLMTokenizer.from_pretrained('facebook/xglm-7.5B')

# Function to convert text to vectors using XGLM
def text_to_vector(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

# Convert text data to feature vectors
X_train = np.array([text_to_vector(text, model, tokenizer) for text in texts_train])
y_train = labels_train

X_test = np.array([text_to_vector(text, model, tokenizer) for text in texts_test])
y_test = labels_test

# Create a pipeline with StandardScaler and SVC
pipeline = make_pipeline(StandardScaler(), SVC())

# Define parameter grid for GridSearchCV
param_grid = {
    'svc__kernel': ['linear','rbf'],
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto']
}

# Implement nested K-fold cross-validation
outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform nested cross-validation and collect results
nested_scores = []

for train_index, test_index in outer_kf.split(X_train):
    X_outer_train, X_outer_test = X_train[train_index], X_train[test_index]
    y_outer_train, y_outer_test = y_train[train_index], y_train[test_index]

    grid_search = GridSearchCV(pipeline, param_grid, cv=inner_kf, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_outer_train, y_outer_train)
    best_model = grid_search.best_estimator_

    y_outer_pred = best_model.predict(X_outer_test)
    f1_outer = f1_score(y_outer_test, y_outer_pred, average='weighted')
    nested_scores.append(f1_outer)

print("Nested cross-validation results:")
print(f"F1 Score: {np.mean(nested_scores)} ± {np.std(nested_scores)}")

# Fit the model on the training dataset
grid_search_final = GridSearchCV(pipeline, param_grid, cv=inner_kf, scoring='f1_weighted', n_jobs=-1)
grid_search_final.fit(X_train, y_train)

# Best hyperparameters
print(f"Best parameters: {grid_search_final.best_params_}")

# Predict and evaluate on the test set
y_pred_test = grid_search_final.predict(X_test)

# Calculate final metrics
accuracy_final = accuracy_score(y_test, y_pred_test)
precision_final = precision_score(y_test, y_pred_test, average='weighted')
recall_final = recall_score(y_test, y_pred_test, average='weighted')
f1_final = f1_score(y_test, y_pred_test, average='weighted')

# Print final evaluation results
print("\nFinal evaluation results on the test set:")
print(f"Accuracy: {accuracy_final}")
print(f"Precision: {precision_final}")
print(f"Recall: {recall_final}")
print(f"F1 Score: {f1_final}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
