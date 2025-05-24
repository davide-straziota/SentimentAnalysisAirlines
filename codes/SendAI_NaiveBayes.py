import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Load the dataset in JSONL format
df_train = pd.read_json('tweet_sentiment.train.jsonl', lines=True)
df_test = pd.read_json('tweet_sentiment.test.jsonl', lines=True)

# Print the first 5 rows of the dataset
print(df_train.head())
print(df_test.head())

# Print the shape of the dataset
print(df_train.shape)
print(df_test.shape)

# Print the columns of the dataset
print(df_train.columns)
print(df_test.columns)

def load_glove_embeddings(file_path):
    embeddings_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
    return embeddings_dict

# Load the GloVe embeddings
glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')

# Print some basic information
print(f"Number of words in vocabulary: {len(glove_embeddings)}")
print(f"Vector dimension: {len(next(iter(glove_embeddings.values())))}")

# Example: Get vector for a word
word = "example"
if word in glove_embeddings:
    print(f"Vector for '{word}': {glove_embeddings[word][:5]}...")  # Print first 5 dimensions

def preprocess_tweet(tweet):
    """
    Preprocess a tweet by:
    1. Converting to lowercase
    2. Removing URLs
    3. Removing mentions (@username)
    4. Removing hashtags (#)
    5. Removing special characters and numbers
    """
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove hashtags
    tweet = re.sub(r'#\w+', '', tweet)
    
    # Remove special characters and numbers
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    
    return tweet

def tweet_to_vector(tweet, glove_embeddings, stop_words=None):
    """
    Convert a tweet to a vector with enhanced weighting
    """
    processed_tweet = preprocess_tweet(tweet)
    tokens = word_tokenize(processed_tweet)
    
    if stop_words:
        tokens = [token for token in tokens if token not in stop_words]
    
    # Calculate word frequencies
    word_freq = {}
    for token in tokens:
        word_freq[token] = word_freq.get(token, 0) + 1
    
    # Create weighted vector
    tweet_vector = np.zeros(100)
    total_words = len(tokens)
    
    for token, freq in word_freq.items():
        if token in glove_embeddings:
            # Enhanced weighting: square the frequency to give more weight to repeated words
            weight = (freq / total_words) ** 2
            tweet_vector += glove_embeddings[token] * weight
    
    # Normalize
    norm = np.linalg.norm(tweet_vector)
    if norm > 0:
        tweet_vector = tweet_vector / norm
    
    return tweet_vector

# Example usage:
# Forza il download dei dati NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Get stop words
stop_words = set(stopwords.words('english'))

# Example tweet
tweet = "Hello! This is a test tweet @user123 #NLP https://example.com"
vector = tweet_to_vector(tweet, glove_embeddings, stop_words)
print(f"Tweet vector shape: {vector.shape}")

# Analyze class distribution
def analyze_class_distribution(df, dataset_name=""):
    # Count class frequencies
    class_counts = df['label'].value_counts()
    total_samples = len(df)
    
    # Calculate percentages
    class_percentages = (class_counts / total_samples * 100).round(2)
    
    print(f"\nClass Distribution for {dataset_name}:")
    print("-" * 40)
    print("Counts:")
    print(class_counts)
    print("\nPercentages:")
    print(class_percentages)
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title(f'Class Distribution - {dataset_name}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()
    
    return class_counts, class_percentages

# Analyze both training and test sets
train_counts, train_percentages = analyze_class_distribution(df_train, "Training Set")
test_counts, test_percentages = analyze_class_distribution(df_test, "Test Set")

# Check for any anomalies
print("\nPotential Anomalies:")
print("-" * 40)

# Check for class imbalance
imbalance_threshold = 0.4  # If any class is less than 40% of the majority class
majority_class_count = max(train_counts)
for class_label, count in train_counts.items():
    if count < majority_class_count * imbalance_threshold:
        print(f"Warning: Class {class_label} is significantly underrepresented ({count} samples, {train_percentages[class_label]}%)")

# Check for missing labels
if df_train['label'].isnull().any():
    print("Warning: There are missing labels in the training set")

# Check for label consistency between train and test
train_labels = set(df_train['label'].unique())
test_labels = set(df_test['label'].unique())
if train_labels != test_labels:
    print(f"Warning: Label mismatch between train and test sets")
    print(f"Labels in train but not in test: {train_labels - test_labels}")
    print(f"Labels in test but not in train: {test_labels - train_labels}")

# 1. Prepara i dati
print("Converting tweets to vectors...")
X_train = np.array([tweet_to_vector(tweet, glove_embeddings, stop_words) for tweet in df_train['text']])
y_train = df_train['label']

# Split training data into train and validation sets
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 2. Scala i dati
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val)

# 1. Converti le etichette in numeri
label_encoder = LabelEncoder()
y_train_numeric = label_encoder.fit_transform(y_train_final)
y_val_numeric = label_encoder.transform(y_val)

# 2. Calcola i pesi delle classi
class_counts = np.bincount(y_train_numeric)
total_samples = len(y_train_numeric)
class_weights = total_samples / (len(np.unique(y_train_numeric)) * class_counts)
class_weights = class_weights / np.sum(class_weights)

# 3. Crea e addestra il modello
nb = ComplementNB(alpha=0.5)
nb.fit(X_train_scaled, y_train_numeric)

# 4. Fai predizioni
y_train_pred = nb.predict(X_train_scaled)
y_val_pred = nb.predict(X_val_scaled)

# 5. Calcola le metriche usando le etichette numeriche
print("\nTraining Set Results:")
print("-" * 40)
train_accuracy = accuracy_score(y_train_numeric, y_train_pred)
train_precision = precision_score(y_train_numeric, y_train_pred, average='weighted')
train_recall = recall_score(y_train_numeric, y_train_pred, average='weighted')
train_f1 = f1_score(y_train_numeric, y_train_pred, average='weighted')

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training Precision: {train_precision:.4f}")
print(f"Training Recall: {train_recall:.4f}")
print(f"Training F1-score: {train_f1:.4f}")

print("\nValidation Set Results:")
print("-" * 40)
val_accuracy = accuracy_score(y_val_numeric, y_val_pred)
val_precision = precision_score(y_val_numeric, y_val_pred, average='weighted')
val_recall = recall_score(y_val_numeric, y_val_pred, average='weighted')
val_f1 = f1_score(y_val_numeric, y_val_pred, average='weighted')

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation F1-score: {val_f1:.4f}")

# 6. Stampa i report di classificazione usando le etichette originali
print("\nTraining Set Classification Report:")
print(classification_report(y_train_final, label_encoder.inverse_transform(y_train_pred)))

print("\nValidation Set Classification Report:")
print(classification_report(y_val, label_encoder.inverse_transform(y_val_pred)))

# 7. Salva il modello e l'encoder
joblib.dump(nb, 'sentiment_model_nb.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
print("\nModel, scaler and label encoder saved")

# 8. Funzione per predire il sentiment di nuovi tweet
def predict_sentiment(tweet):
    """
    Predict the sentiment of a new tweet.
    """
    # Convert tweet to vector
    tweet_vector = tweet_to_vector(tweet, glove_embeddings, stop_words)
    # Scale the vector
    tweet_vector_scaled = scaler.transform(tweet_vector.reshape(1, -1))
    # Predict
    prediction_numeric = nb.predict(tweet_vector_scaled)
    # Convert back to original label
    prediction = label_encoder.inverse_transform(prediction_numeric)
    return prediction[0]

# Example usage
test_tweet = "I really enjoyed this movie, it was fantastic!"
prediction = predict_sentiment(test_tweet)
print(f"\nExample prediction for '{test_tweet}':")
print(f"Predicted sentiment: {prediction}")

# 1. Convert test tweets to vectors
print("Converting test tweets to vectors...")
X_test = np.array([tweet_to_vector(tweet, glove_embeddings, stop_words) for tweet in df_test['text']])
y_test = df_test['label']

# 2. Scale the test features using the same scaler
X_test_scaled = scaler.transform(X_test)

# 3. Make predictions on all sets
print("Making predictions...")
y_test_pred = nb.predict(X_test_scaled)

# 4. Calculate and print metrics
# Calculate accuracies
y_test_numeric = label_encoder.transform(y_test)
test_accuracy = accuracy_score(y_test_numeric, y_test_pred)

# Test Set Metrics
print("\nTest Set Results:")
print("-" * 40)
test_precision = precision_score(y_test_numeric, y_test_pred, average='weighted')
test_recall = recall_score(y_test_numeric, y_test_pred, average='weighted')
test_f1 = f1_score(y_test_numeric, y_test_pred, average='weighted')

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-score: {test_f1:.4f}")

# Detailed classification reports
print("\nTest Set Classification Report:")
print(classification_report(y_test, label_encoder.inverse_transform(y_test_pred)))

# Confusion Matrices
plt.figure(figsize=(15, 5))

# Training Set Confusion Matrix
plt.subplot(1, 3, 1)
cm_train = confusion_matrix(y_train_numeric, y_train_pred)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Training Set')
plt.xlabel('Predicted')
plt.ylabel('True')

# Validation Set Confusion Matrix
plt.subplot(1, 3, 2)
cm_val = confusion_matrix(y_val_numeric, y_val_pred)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Validation Set')
plt.xlabel('Predicted')
plt.ylabel('True')

# Test Set Confusion Matrix
plt.subplot(1, 3, 3)
cm_test = confusion_matrix(y_test_numeric, y_test_pred)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Test Set')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()

# 5. Analyze some example predictions
print("\nExample Predictions from Test Set:")
print("-" * 40)
# Get some random examples from test set
n_examples = 5
random_indices = np.random.choice(len(df_test), n_examples, replace=False)

# To do: 
# New learning methods
# imbalanced data
# Data augmentation
