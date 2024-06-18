import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import BertModel, BertTokenizer

# Load the datasets
file_path_train = '/home/u161198/new_env/Datasets/EXIST_2021_Dataset/training/EXIST2021_training.tsv'
file_path_test = '/home/u161198/new_env/Datasets/EXIST_2021_Dataset/test/EXIST2021_test_labeled.tsv'

Sexism_complete_train = pd.read_csv(file_path_train, delimiter='\t')
Sexism_complete_test = pd.read_csv(file_path_test, delimiter='\t')

# Filter datasets to include only Spanish tweets
Sexism_train = Sexism_complete_train[(Sexism_complete_train['language'] == 'es') & (Sexism_complete_train['source'] == 'twitter')]
Sexism_test = Sexism_complete_test[(Sexism_complete_test['language'] == 'es') & (Sexism_complete_test['source'] == 'twitter')]

# BETO model and tokenizer
BETO = 'dccuchile/bert-base-spanish-wwm-cased'
tokenizer = BertTokenizer.from_pretrained(BETO)
model = BertModel.from_pretrained(BETO)

# Feature extraction function using BETO
def extract_features(text_list):
    input_ids_list = [tokenizer.encode(text, add_special_tokens=True) for text in text_list]
    input_ids_list = [torch.tensor(ids) for ids in input_ids_list]
    input_ids_list = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    attention_mask = [[int(token_id > 0) for token_id in input_ids] for input_ids in input_ids_list]

    input_ids_tensor = torch.tensor(input_ids_list)
    attention_mask_tensor = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids_tensor, attention_mask=attention_mask_tensor)[0]
    return last_hidden_states.mean(1).cpu().numpy()

# Separate features and labels for train and test sets
texts_train = Sexism_train['text'].tolist()
labels_train = Sexism_train['task1'].tolist()

texts_test = Sexism_test['text'].tolist()
labels_test = Sexism_test['task1'].tolist()

# Extract features using BETO
X_train = extract_features(texts_train)
X_test = extract_features(texts_test)

y_train = np.array(labels_train)
y_test = np.array(labels_test)

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

for train_index, test_index in outer_kf.split(X_train):
    X_outer_train, X_outer_test = X_train[train_index], X_train[test_index]
    y_outer_train, y_outer_test = y_train[train_index], y_train[test_index]

    pipeline = make_pipeline(StandardScaler(), SVC())

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
best_model = grid_search_final.best_estimator_
y_pred_test = best_model.predict(X_test)

# Calculate final metrics
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test, average='weighted')
recall_test = recall_score(y_test, y_pred_test, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')

# Print final evaluation results
print("\nTest set results:")
print(f"Accuracy: {accuracy_test}")
print(f"Precision: {precision_test}")
print(f"Recall: {recall_test}")
print(f"F1 Score: {f1_test}")

# Detailed classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
