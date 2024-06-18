
import torch
from transformers import XGLMModel, XGLMTokenizer
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Reading the training dataset
file_path_train = '/home/u161198/new_env/Datasets/EXIST_2021_Dataset/training/EXIST2021_training.tsv'
Sexism_complete_train = pd.read_csv(file_path_train, delimiter='\t')
Sexism_train = Sexism_complete_train[(Sexism_complete_train['language'] == 'es') & (Sexism_complete_train['source'] == 'twitter')]

# Reading the testing dataset
file_path_test = '/home/u161198/new_env/Datasets/EXIST_2021_Dataset/test/EXIST2021_test_labeled.tsv'
Sexism_complete_test = pd.read_csv(file_path_test, delimiter='\t')
Sexism_test = Sexism_complete_test[(Sexism_complete_test['language'] == 'es') & (Sexism_complete_test['source'] == 'twitter')]

texts_train = Sexism_train['text']
labels_train = Sexism_train['task1']
texts_test = Sexism_test['text']
labels_test = Sexism_test['task1']

# Load the model and tokenizer
model_name = "facebook/xglm-564M"
tokenizer = XGLMTokenizer.from_pretrained(model_name)
model = XGLMModel.from_pretrained(model_name)

def text_to_vector(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

# Convert text data to feature vectors for training and testing sets 
X_train = np.array([text_to_vector(text, model, tokenizer) for text in texts_train])
y_train = labels_train.values

X_test = np.array([text_to_vector(text, model, tokenizer) for text in texts_test])
y_test = labels_test.values

# Create a pipeline with StandardScaler and SVC
pipeline = make_pipeline(StandardScaler(), SVC())

# Define parameter grid for GridSearchCV
param_grid = {
    'svc__kernel': ['linear', 'rbf'],
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto']
}

# Implement nested K-fold cross-validation
outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform nested cross-validation and collect results
nested_scores = []

for train_index, test_index in outer_kf.split(X):
    X_outer_train, X_outer_test = X[train_index], X[test_index]
    y_outer_train, y_outer_test = y[train_index], y[test_index]

    grid_search = GridSearchCV(pipeline, param_grid, cv=inner_kf, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_outer_train, y_outer_train)
    best_model = grid_search.best_estimator_

    y_outer_pred = best_model.predict(X_outer_test)
    f1_outer = f1_score(y_outer_test, y_outer_pred, average='weighted')
    nested_scores.append(f1_outer)

print("Nested cross-validation results:")
print(f"F1 Score: {np.mean(nested_scores)} ± {np.std(nested_scores)}")

# Fit the model on the entire training dataset
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

# Detailed classification report and confusion matrix 
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

