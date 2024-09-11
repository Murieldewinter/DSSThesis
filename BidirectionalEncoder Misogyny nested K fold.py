
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import msoffcrypto
import io

# Decrypt and read the training dataset
file_path = '/home/u161198/new_env/Datasets/es_AMI_TrainingSet_NEW (2).xlsx'
password = 

file = msoffcrypto.OfficeFile(open(file_path, 'rb'))
file.load_key(password=password)

decrypted_stream = io.BytesIO()
file.decrypt(decrypted_stream)

decrypted_stream.seek(0)
Misogyny_train = pd.read_excel(decrypted_stream, engine='openpyxl')

# Read the testing dataset
file_path = '/home/u161198/new_env/Datasets/es_AMI_TestSet.tsv'
Misogyny_test = pd.read_csv(file_path, delimiter='\t')

# Extract texts and labels
texts_train = Misogyny_train['tweet'].tolist()
labels_train = Misogyny_train['misogynous'].tolist()

texts_test = Misogyny_test['tweet'].tolist()
labels_test = Misogyny_test['misogynous'].tolist()

# BETO model and tokenizer
BETO = 'dccuchile/bert-base-spanish-wwm-cased'
tokenizer = BertTokenizer.from_pretrained(BETO)
model = BertModel.from_pretrained(BETO)

# Function to extract features using BETO
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

# Extract features for training and testing sets
X_train = extract_features(texts_train)
X_test = extract_features(texts_test)
y_train = np.array(labels_train)
y_test = np.array(labels_test)

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

# Fit the final model on the entire training dataset
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

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))
