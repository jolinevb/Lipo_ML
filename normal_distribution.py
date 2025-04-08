# Data loading
from worcliver.load_data import load_data
from sklearn.preprocessing import robust_scale
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC


data = load_data()
X_scaled = robust_scale(data[data.columns[1:]])
data[data.columns[1:]] = X_scaled


def normal_distribution(data):
    '''
    This function checks the normal distribution of features in the dataset and performs statistical tests.
    '''
    benign = data[data['label'] == 'benign']
    malignant = data[data['label'] == 'malignant']

    normally_distributed_features = []  # List to store normally distributed features
    statistically_significant_features = []  # List to store statistically significant features
    statistically_significant_features_number = []  # List to store statistically significant features number

    # Loop through all features (excluding the label column)

    for feature_number, feature in enumerate(data.columns[1:], start=1):
        selected_feature_benign = benign[feature]
        selected_feature_malignant = malignant[feature]

        # Shapiro-Wilk test for normality
        stat_benign, p_benign = shapiro(selected_feature_benign)
        stat_malignant, p_malignant = shapiro(selected_feature_malignant)

        if p_benign > 0.05 and p_malignant > 0.05:
            # print(f"Feature '{feature}' is normally distributed")
            normally_distributed_features.append(feature)

            # Perform t-test for statistical significance
            stat, p_value = stats.ttest_ind(selected_feature_benign, selected_feature_malignant)

            # Correct for multiple comparisons (Bonferroni correction)
            p_value *= len(normally_distributed_features)
                           
            if p_value < 0.05:
                print(f"Feature '{feature}' is statistically significant (p-value: {p_value:.4f})")

                # Save the feature number
                statistically_significant_features.append(feature)
                statistically_significant_features_number.append(feature_number)
    return benign, malignant, normally_distributed_features, statistically_significant_features, statistically_significant_features_number

# call the function
[benign, malignant,
 normally_distributed_features,
 statistically_significant_features,
 statistically_significant_features_number] = normal_distribution(data)

# Plot scatter of statistically significant features
fig, axis = plt.subplots(1, 3, figsize=(15, 5))

axis[0].scatter(benign[statistically_significant_features[0]], benign[statistically_significant_features[1]], color='blue', label='Benign', s=3)
axis[0].scatter(malignant[statistically_significant_features[0]], malignant[statistically_significant_features[1]], color='red', label='Malignant', s=3)
axis[0].set_xlabel(f"Feature number {statistically_significant_features_number[0]}")
axis[0].set_ylabel(f"Feature number {statistically_significant_features_number[1]}")
axis[0].legend()

axis[1].scatter(benign[statistically_significant_features[0]], benign[statistically_significant_features[2]], color='blue', label='Benign', s=3)
axis[1].scatter(malignant[statistically_significant_features[0]], malignant[statistically_significant_features[2]], color='red', label='Malignant', s=3)
axis[1].set_xlabel(f"Feature number {statistically_significant_features_number[0]}")
axis[1].set_ylabel(f"Feature number {statistically_significant_features_number[2]}")
axis[1].legend()

axis[2].scatter(benign[statistically_significant_features[1]], benign[statistically_significant_features[2]], color='blue', label='Benign', s=3)
axis[2].scatter(malignant[statistically_significant_features[1]], malignant[statistically_significant_features[2]], color='red', label='Malignant', s=3)
axis[2].set_xlabel(f"Feature number {statistically_significant_features_number[1]}")
axis[2].set_ylabel(f"Feature number {statistically_significant_features_number[2]}")
axis[2].legend()

plt.show()

# Train a logistic regression model with the statistically significant features

# Select the statistically significant features
X = data[statistically_significant_features]
y = data['label'].apply(lambda x: 1 if x == 'malignant' else 0)  # Convert labels to binary (0 for benign, 1 for malignant)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the logistic regression model
models = [SVC(), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()]
for model_instance in models:
    # Fit the model on the training data
    model_instance.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = model_instance.predict(X_test)
    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training model: {model_instance.__class__.__name__}, Model Accuracy: {accuracy:.4f}")

