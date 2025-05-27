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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def load_glove_embeddings(file_path):
    embeddings_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
    return embeddings_dict

def preprocess_tweet(tweet):
    """Preprocess a tweet."""
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    return tweet

def tweet_to_vector(tweet, glove_embeddings, stop_words=None):
    """Convert a tweet to a vector."""
    processed_tweet = preprocess_tweet(tweet)
    tokens = word_tokenize(processed_tweet)
    
    if stop_words:
        tokens = [token for token in tokens if token not in stop_words]
    
    word_freq = {}
    for token in tokens:
        word_freq[token] = word_freq.get(token, 0) + 1
    
    tweet_vector = np.zeros(100)
    total_words = len(tokens)
    
    for token, freq in word_freq.items():
        if token in glove_embeddings:
            weight = (freq / total_words) ** 2
            tweet_vector += glove_embeddings[token] * weight
    
    norm = np.linalg.norm(tweet_vector)
    if norm > 0:
        tweet_vector = tweet_vector / norm
    
    return tweet_vector

def apply_imbalanced_technique(X, y, technique='smote'):
    """Apply different imbalanced learning techniques."""
    if technique == 'smote':
        sampler = SMOTE(random_state=42)
    elif technique == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif technique == 'smoteenn':
        sampler = SMOTEENN(random_state=42)
    elif technique == 'smotetomek':
        sampler = SMOTETomek(random_state=42)
    elif technique == 'under':
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError(f"Unknown technique: {technique}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def main():
    # Load the dataset
    df_train = pd.read_json('tweet_sentiment.train.jsonl', lines=True)
    df_test = pd.read_json('tweet_sentiment.test.jsonl', lines=True)

    # Print dataset information
    print("\nDataset Information:")
    print("-" * 40)
    print("Training set shape:", df_train.shape)
    print("Test set shape:", df_test.shape)
    
    # Analyze class distribution
    def analyze_class_distribution(df, dataset_name=""):
        class_counts = df['label'].value_counts()
        total_samples = len(df)
        class_percentages = (class_counts / total_samples * 100).round(2)
        
        print(f"\nClass Distribution for {dataset_name}:")
        print("-" * 40)
        print("Counts:")
        print(class_counts)
        print("\nPercentages:")
        print(class_percentages)
        
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
    
    # Load GloVe embeddings
    print("\nLoading GloVe embeddings...")
    glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')
    print(f"Number of words in vocabulary: {len(glove_embeddings)}")
    print(f"Vector dimension: {len(next(iter(glove_embeddings.values())))}")
    
    # Convert tweets to vectors
    print("\nConverting tweets to vectors...")
    stop_words = set(stopwords.words('english'))
    X_train = np.array([tweet_to_vector(tweet, glove_embeddings, stop_words) 
                       for tweet in df_train['text']])
    y_train = df_train['label']
    
    # Split training data
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_val_scaled = scaler.transform(X_val)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_numeric = label_encoder.fit_transform(y_train_final)
    y_val_numeric = label_encoder.transform(y_val)
    
    # List of techniques to try
    techniques = ['smote', 'adasyn', 'smoteenn', 'smotetomek', 'under']
    results = {}
    
    # Try different imbalanced learning techniques
    for technique in techniques:
        print(f"\n{'='*50}")
        print(f"Trying {technique.upper()} technique...")
        print(f"{'='*50}")
        
        # Apply the technique
        X_resampled, y_resampled = apply_imbalanced_technique(X_train_scaled, y_train_numeric, technique)
        
        print(f"\nClass distribution after {technique}:")
        print(Counter(y_resampled))
        
        # Calculate class weights
        class_counts = np.bincount(y_resampled)
        total_samples = len(y_resampled)
        class_weights = total_samples / (len(np.unique(y_resampled)) * class_counts)
        class_weights = class_weights / np.sum(class_weights)
        
        # Train XGBoost model
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=len(np.unique(y_resampled)),
            scale_pos_weight=class_weights,
            eval_metric=['mlogloss', 'merror'],
            random_state=42,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        
        # Fit the model
        xgb_model.fit(
            X_resampled, 
            y_resampled,
            eval_set=[(X_val_scaled, y_val_numeric)],
            verbose=True
        )
        
        # Make predictions on validation set
        y_val_pred = xgb_model.predict(X_val_scaled)
        
        # Calculate metrics
        val_accuracy = accuracy_score(y_val_numeric, y_val_pred)
        val_precision = precision_score(y_val_numeric, y_val_pred, average='weighted')
        val_recall = recall_score(y_val_numeric, y_val_pred, average='weighted')
        val_f1 = f1_score(y_val_numeric, y_val_pred, average='weighted')
        
        print(f"\nValidation Results for {technique.upper()}:")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"Recall: {val_recall:.4f}")
        print(f"F1-score: {val_f1:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_val, label_encoder.inverse_transform(y_val_pred)))
        
        # Store results
        results[technique] = {
            'model': xgb_model,
            'val_metrics': {
                'accuracy': val_accuracy,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1
            },
            'val_predictions': y_val_pred
        }
        
        # Plot confusion matrix for validation set
        plt.figure(figsize=(10, 8))
        cm_val = confusion_matrix(y_val_numeric, y_val_pred)
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Validation Set ({technique.upper()})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(xgb_model.evals_result()['validation_0']['mlogloss'], label='Validation Loss')
        plt.title(f'XGBoost Learning Curve - {technique.upper()}')
        plt.xlabel('Iteration')
        plt.ylabel('Multi-class Log Loss')
        plt.legend()
        plt.show()
    
    # Find best technique
    best_technique = max(results.keys(), key=lambda k: results[k]['val_metrics']['f1'])
    best_model = results[best_technique]['model']
    best_f1 = results[best_technique]['val_metrics']['f1']
    
    print(f"\n{'='*50}")
    print("Summary of all techniques:")
    print(f"{'='*50}")
    for technique in techniques:
        metrics = results[technique]['val_metrics']
        print(f"\n{technique.upper()}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
    
    print(f"\nBest technique: {best_technique}")
    print(f"Best validation F1-score: {best_f1:.4f}")
    
    # Test set evaluation with best model
    print("\nEvaluating best model on test set...")
    X_test = np.array([tweet_to_vector(tweet, glove_embeddings, stop_words) 
                      for tweet in df_test['text']])
    X_test_scaled = scaler.transform(X_test)
    y_test = df_test['label']
    y_test_numeric = label_encoder.transform(y_test)
    y_test_pred = best_model.predict(X_test_scaled)
    
    print("\nTest Set Results:")
    print("-" * 40)
    test_accuracy = accuracy_score(y_test_numeric, y_test_pred)
    test_precision = precision_score(y_test_numeric, y_test_pred, average='weighted')
    test_recall = recall_score(y_test_numeric, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test_numeric, y_test_pred, average='weighted')

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, label_encoder.inverse_transform(y_test_pred)))
    
    # Final confusion matrix for test set
    plt.figure(figsize=(10, 8))
    cm_test = confusion_matrix(y_test_numeric, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Test Set (Best Model: {best_technique.upper()})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Feature Importance Plot for best model
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(best_model, max_num_features=20)
    plt.title(f'Feature Importance (Best Model: {best_technique.upper()})')
    plt.show()
    
    # Save the model and components
    joblib.dump(best_model, 'sentiment_model_xgb_imbalanced.joblib')
    joblib.dump(scaler, 'scaler_xgb_imbalanced.joblib')
    joblib.dump(label_encoder, 'label_encoder_xgb_imbalanced.joblib')
    print("\nModel, scaler and label encoder saved")

if __name__ == "__main__":
    main() 