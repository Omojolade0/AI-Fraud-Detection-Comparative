# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neighbors, metrics

# Load the dataset
data = pd.read_csv('fraudDataset.csv')

# Handling missing values
data.dropna(inplace=True)

# Removing duplicates
data.drop_duplicates(inplace=True)

# Encoding categorical variables
label_encoder = LabelEncoder()
for column in ['gender', 'merchant', 'city', 'category', 'state', 'job']:
    data[f'{column}_encoded'] = label_encoder.fit_transform(data[column])

# Scaling numerical variables
scaler = StandardScaler()
for column in ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']:
    data[f'{column}_scaled'] = scaler.fit_transform(data[[column]])

# Extracting hour from transaction date and time
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['Hours'] = data['trans_date_trans_time'].dt.hour

# Selecting features
selected_features = ['gender_encoded', 'merchant_encoded', 'city_encoded', 'category_encoded', 'state_encoded', 'job_encoded',
                     'amt_scaled', 'lat_scaled', 'long_scaled', 'city_pop_scaled', 'merch_lat_scaled', 'merch_long_scaled', 'Hours']

# Handling missing columns
missing_columns = [col for col in selected_features if col not in data.columns]
print("Missing columns:", missing_columns)

# Filtering available selected features
selected_features = [col for col in selected_features if col in data.columns]

# Resampling using SMOTE
smote = SMOTE(random_state=42)
features_resampled, target_resampled = smote.fit_resample(data[selected_features], data['is_fraud'])
data_resampled = pd.concat([features_resampled, target_resampled], axis=1)
print(data_resampled.shape)

X = data_resampled[selected_features]
Y = data_resampled['is_fraud']

# First dataset (10%)
percentage = 0.1
X_partial = X[:int(percentage * len(X))]
Y_partial = Y[:int(percentage * len(Y))]
X_partial_train, X_partial_test, Y_partial_train, Y_partial_test = train_test_split(X_partial, Y_partial, test_size=0.2, random_state=42)

# KNN model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
knn.fit(X_partial_train, Y_partial_train)
knn_predictions_first = knn.predict(X_partial_test)
accuracy_knn_first = metrics.accuracy_score(Y_partial_test, knn_predictions_first)
print("Accuracy (KNN First):", accuracy_knn_first)

# Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_partial_train, Y_partial_train)
rf_predictions_first = rf_classifier.predict(X_partial_test)
accuracy_rf_first = metrics.accuracy_score(Y_partial_test, rf_predictions_first)
print("Accuracy (Random Forest First):", accuracy_rf_first)

# SVM model
model = svm.SVC()
model.fit(X_partial_train, Y_partial_train)
svm_predictions_first = model.predict(X_partial_test)
accuracy_svm_first = accuracy_score(Y_partial_test, svm_predictions_first)
print("Accuracy (SVM First):", accuracy_svm_first)

# Second dataset (20%)
percentage_second = 0.2
X_second = X[:int(percentage_second * len(X))]
Y_second = Y[:int(percentage_second * len(Y))]
X_second_train, X_second_test, Y_second_train, Y_second_test = train_test_split(X_second, Y_second, test_size=0.2, random_state=42)

# SVM model on second dataset
model = svm.SVC()
model.fit(X_second_train, Y_second_train)
svm_predictions_second = model.predict(X_second_test)
accuracy_svm_second = accuracy_score(Y_second_test, svm_predictions_second)
print("Accuracy (SVM Second):", accuracy_svm_second)

# Third dataset (50%)
percentage_third = 0.5
X_third = X[:int(percentage_third * len(X))]
Y_third = Y[:int(percentage_third * len(Y))]
X_third_train, X_third_test, Y_third_train, Y_third_test = train_test_split(X_third, Y_third, test_size=0.2, random_state=42)

# SVM model on third dataset
model = svm.SVC()
model.fit(X_third_train, Y_third_train)
svm_predictions_third = model.predict(X_third_test)
accuracy_svm_third = accuracy_score(Y_third_test, svm_predictions_third)
print("Accuracy (SVM Third):", accuracy_svm_third)

# ON full dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

knn = neighbors.KNeighborsClassifier(n_neighbors= 25, weights='uniform')
knn.fit(X_train, Y_train)
knn_predictions = knn.predict(X_test)
accuracy_knn = metrics.accuracy_score(Y_test, knn_predictions)
print("Accuracy:", accuracy_knn)

#KNN visualization
conf_matrix = confusion_matrix(Y_test, knn_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, Y_train)
rf_predictions = rf_classifier.predict(X_test)
accuracy_rf = metrics.accuracy_score(Y_test, rf_predictions)
print(accuracy_rf)

#RF visulaization
# RF visualization
correct_counts = sum(1 for true, pred in zip(Y_test, rf_predictions) if true == pred)
incorrect_counts = len(Y_test) - correct_counts

plt.figure(figsize=(8, 6))
plt.bar(['Correct Predictions', 'Incorrect Predictions'], [correct_counts, incorrect_counts], color=['green', 'red'])
plt.xlabel('Prediction Outcome')
plt.ylabel('Count')
plt.title('Count of Correct and Incorrect Predictions for Random Forest')
plt.show()



# Data visualization before after SMOTE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(data['is_fraud'], color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Target Feature (Before SMOTE)')
plt.xlabel('is_fraud')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(target_resampled, color='salmon', edgecolor='black', alpha=0.7)
plt.title('Distribution of Target Feature (After SMOTE)')
plt.xlabel('is_fraud')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
