import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Load and preprocess data
print("Loading data...")
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Print data distribution before oversampling
labels = np.asarray(data_dict['labels'])
label_counts = Counter(labels)
print("\nClass distribution before oversampling:")
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")

# Standardize the data
print("\nPreprocessing data...")
max_length = max(len(item) for item in data_dict['data'])
standardized_data = [np.pad(item, (0, max_length - len(item)), 'constant') if len(item) < max_length else item[:max_length] for item in data_dict['data']]
data = np.asarray(standardized_data)

# Scale the features
scaler = RobustScaler()
data_scaled = scaler.fit_transform(data)

# Apply initial random oversampling to ensure at least 2 samples per class
print("\nApplying initial random oversampling...")
initial_oversampler = RandomOverSampler(random_state=42)
data_initial_ros, labels_initial_ros = initial_oversampler.fit_resample(data_scaled, labels)

print("\nClass distribution after initial oversampling:")
for label, count in Counter(labels_initial_ros).items():
    print(f"Label {label}: {count} samples")

# Split the data
print("\nSplitting data...")
x_train, x_test, y_train, y_test = train_test_split(
    data_initial_ros, labels_initial_ros,
    test_size=0.2, shuffle=True,
    stratify=labels_initial_ros,
    random_state=42
)

# Apply additional random oversampling on training data
print("\nApplying additional random oversampling...")
random_oversampler = RandomOverSampler(random_state=42)
x_train_ros, y_train_ros = random_oversampler.fit_resample(x_train, y_train)

print("\nClass distribution after additional random oversampling:")
for label, count in Counter(y_train_ros).items():
    print(f"Label {label}: {count} samples")

# Apply SMOTE
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_ros, y_train_ros)

print("\nClass distribution after SMOTE:")
for label, count in Counter(y_train_resampled).items():
    print(f"Label {label}: {count} samples")

# Create and train the classifier
print("\nTraining model...")
classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

# Perform cross-validation
print("\nPerforming cross-validation...")
cv_scores = cross_val_score(classifier, x_train_resampled, y_train_resampled, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

classifier.fit(x_train_resampled, y_train_resampled)

# Evaluate model
print("\nEvaluating model...")
y_predict = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f'\nAccuracy: {accuracy * 100:.2f}%')

# Get prediction probabilities
y_proba = classifier.predict_proba(x_test)
max_probs = np.max(y_proba, axis=1)
print(f"\nAverage prediction confidence: {np.mean(max_probs):.2f}")
print(f"Min prediction confidence: {np.min(max_probs):.2f}")
print(f"Max prediction confidence: {np.max(max_probs):.2f}")

print("\nDetailed classification report:")
print(classification_report(y_test, y_predict))

print("\nSaving model...")
f = open('model.p', 'wb')
pickle.dump({
    'scaler': scaler,
    'classifier': classifier
}, f)
f.close()
print("Model saved successfully!")
