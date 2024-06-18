import numpy as np
import pandas as pd
import gensim
from gensim.models import KeyedVectors
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load the datasets
file_path_train = '/home/u161198/new_env/Datasets/EXIST_2021_Dataset/training/EXIST2021_training.tsv'
Sexism_complete_train = pd.read_csv(file_path_train, delimiter='\t')
file_path_test = '/home/u161198/new_env/Datasets/EXIST_2021_Dataset/test/EXIST2021_test_labeled.tsv'
Sexism_complete_test = pd.read_csv(file_path_test, delimiter='\t')

# Filter datasets to include only Spanish tweets
Sexism_train = Sexism_complete_train[(Sexism_complete_train['language'] == 'es') & (Sexism_complete_train['source'] == 'twitter')]
Sexism_test = Sexism_complete_test[(Sexism_complete_test['language'] == 'es') & (Sexism_complete_test['source'] == 'twitter')]

texts_train = Sexism_train['text']
labels_train = Sexism_train['task1']

texts_test = Sexism_test['text']
labels_test = Sexism_test['task1']

# Preprocess the text data and convert to feature vectors
def text_to_vector(text, model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Load pre-trained SBWCE Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format('/home/u161198/new_env/Datasets/SBW-vectors-300-min5.bin.gz', binary=True)

# Convert texts to embeddings for train and test datasets
X_train = np.array([text_to_vector(text, word2vec_model) for text in texts_train])
X_test = np.array([text_to_vector(text, word2vec_model) for text in texts_test])
y_train = labels_train.values
y_test = labels_test.values

# Create a pipeline with StandardScaler and SVC
pipeline = make_pipeline(StandardScaler(), SVC())

# Define parameter grid for GridSearchCV
param_grid = {
    'svc__kernel': ['linear', 'rbf'],
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto']
}

# Outer KFold for nested cross-validation
outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Collect results for each fold
nested_scores = []

for train_index, test_index in outer_kf.split(X_train, y_train):
    X_outer_train, X_outer_test = X_train[train_index], X_train[test_index]
    y_outer_train, y_outer_test = y_train[train_index], y_train[test_index]

    # Inner KFold for hyperparameter tuning
    inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=inner_kf, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_outer_train, y_outer_train)
    best_model = grid_search.best_estimator_

    y_outer_pred = best_model.predict(X_outer_test)
    f1_outer = f1_score(y_outer_test, y_outer_pred, average='weighted')
    nested_scores.append(f1_outer)

    print(f"Fold completed with F1 Score: {f1_outer}")

print("Nested cross-validation results:")
print(f"F1 Score: {np.mean(nested_scores)} ± {np.std(nested_scores)}")

# Fit the final model on the entire training dataset
grid_search_final = GridSearchCV(pipeline, param_grid, cv=inner_kf, scoring='f1_weighted', n_jobs=-1)
grid_search_final.fit(X_train, y_train)

# Best hyperparameters
print(f"Best parameters: {grid_search_final.best_params_}")

# Predict and evaluate on the test set using the best parameters
best_model = grid_search_final.best_estimator_
y_pred_test = best_model.predict(X_test)

# Calculate final metrics
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test, average='weighted')
recall_test = recall_score(y_test, y_pred_test, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')

# Print final evaluation results and confusion matrix 
print("\nTest set results:")
print(f"Accuracy: {accuracy_test}")
print(f"Precision: {precision_test}")
print(f"Recall: {recall_test}")
print(f"F1 Score: {f1_test}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

