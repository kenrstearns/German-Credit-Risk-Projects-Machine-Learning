# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Step 2: Load ARFF File and Prepare the Data
# Load the ARFF file
data = arff.loadarff('dataset_31_credit-g.arff')
df = pd.DataFrame(data[0])

# Convert byte strings to normal strings if needed
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Define features and target
X = df.drop('class', axis=1)  # Assuming 'class' is the target column
y = df['class']

# Encode target variable (assuming 'good' or 'bad' labels)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 3: Data Preprocessing
# Example of encoding categorical variables
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Ensure the same columns in train and test set after encoding
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Model Selection and Training
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42)
}

# Train each model
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    print(f'{model_name} trained successfully.')

# Step 5: Hyperparameter Tuning Example (Random Forest)
random_forest_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Randomized search for hyperparameter tuning
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=random_forest_params,
    n_iter=20,
    scoring='f1',
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train)
best_rf_model = random_search.best_estimator_
print("Best Parameters for Random Forest:", random_search.best_params_)

# Step 6: Evaluate Model on Test Set
y_pred = best_rf_model.predict(X_test_scaled)
y_proba = best_rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nTuned Random Forest Performance:")
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')
print(f'Recall: {recall_score(y_test, y_pred):.4f}')
print(f'F1-Score: {f1_score(y_test, y_pred):.4f}')
print(f'ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Bad', 'Good'], rotation=45)
plt.yticks(tick_marks, ['Bad', 'Good'])
plt.tight_layout()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Step 7: Feature Importance for Random Forest
importance = best_rf_model.feature_importances_
feature_names = X_train.columns

# Sort and plot the top 10 feature importances
indices = np.argsort(importance)[-10:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importance[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importance Score')
plt.title('Top 10 Feature Importances - Random Forest')
plt.show()

# Save Results for New Predictions (Optional)
# new_data = pd.read_csv('new_credit_applicants.csv')
# Preprocess and predict using best_rf_model...
