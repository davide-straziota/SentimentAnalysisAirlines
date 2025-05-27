import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import ssl
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import random
from deep_translator import GoogleTranslator
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from imblearn.over_sampling import SMOTE
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
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class DataAugmenter:
    def __init__(self, use_advanced_augmentations=False):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Precompute synonyms dictionary for faster access
        self.synonyms_dict = {}
        print("Precomputing synonyms dictionary...")
        for synset in wordnet.all_synsets():
            for lemma in synset.lemmas():
                word = lemma.name().replace('_', ' ')
                if word not in self.synonyms_dict:
                    self.synonyms_dict[word] = []
                for synonym in synset.lemma_names():
                    synonym = synonym.replace('_', ' ')
                    if synonym != word and synonym not in self.synonyms_dict[word]:
                        self.synonyms_dict[word].append(synonym)
        
        # Initialize basic augmenters
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
        
        # Initialize advanced augmenters only if requested and available
        self.context_aug = None
        self.char_aug = None
        if use_advanced_augmentations:
            try:
                import torch
                self.context_aug = naw.ContextualWordEmbsAug(
                    model_path='bert-base-uncased',
                    action="substitute"
                )
                self.char_aug = nac.KeyboardAug()
                print("Advanced augmentations (BERT-based) enabled")
            except ImportError:
                print("Warning: PyTorch not found. Advanced augmentations disabled.")
                print("To enable advanced augmentations, install PyTorch following: https://pytorch.org/get-started/locally/")

    def get_synonyms(self, word):
        """Get synonyms for a word using precomputed dictionary."""
        return self.synonyms_dict.get(word, [])

    def augment_single_text(self, text, augmentations=None):
        """Apply multiple augmentation techniques to a single text."""
        if augmentations is None:
            augmentations = ['synonym', 'deletion', 'swap']
            
        augmented_texts = [text]
        
        if 'synonym' in augmentations:
            augmented_texts.append(self.synonym_replacement(text))
            
        if 'deletion' in augmentations:
            augmented_texts.append(self.random_deletion(text))
            
        if 'swap' in augmentations:
            augmented_texts.append(self.random_swap(text))
            
        return augmented_texts

    def synonym_replacement(self, text, n=1):
        """Replace n words with their synonyms."""
        words = word_tokenize(text)
        n = min(n, len(words))
        new_words = words.copy()
        
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) > 0:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
                
        return ' '.join(new_words)

    def back_translation(self, text, target_lang='fr'):
        """Perform back translation augmentation using deep-translator."""
        try:
            # Translate to target language
            translator = GoogleTranslator(source='en', target=target_lang)
            translated = translator.translate(text)
            
            # Translate back to English
            translator = GoogleTranslator(source=target_lang, target='en')
            back_translated = translator.translate(translated)
            
            return back_translated
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text

    def random_deletion(self, text, p=0.1):
        """Randomly delete words with probability p."""
        words = word_tokenize(text)
        if len(words) == 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
                
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return words[rand_int]
            
        return ' '.join(new_words)

    def random_swap(self, text, n=1):
        """Randomly swap n pairs of words."""
        words = word_tokenize(text)
        n = min(n, len(words)-1)
        new_words = words.copy()
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
        return ' '.join(new_words)

    def augment_text(self, text, augmentations=None):
        """Apply multiple augmentation techniques to the text."""
        if augmentations is None:
            augmentations = ['synonym', 'back_translation', 'deletion', 'swap']
            
        augmented_texts = [text]
        
        if 'synonym' in augmentations:
            augmented_texts.append(self.synonym_replacement(text))
            
        if 'back_translation' in augmentations:
            augmented_texts.append(self.back_translation(text))
            
        if 'deletion' in augmentations:
            augmented_texts.append(self.random_deletion(text))
            
        if 'swap' in augmentations:
            augmented_texts.append(self.random_swap(text))
            
        if 'nlpaug_synonym' in augmentations and self.synonym_aug:
            augmented_texts.append(self.synonym_aug.augment(text)[0])
            
        if 'nlpaug_context' in augmentations and self.context_aug:
            augmented_texts.append(self.context_aug.augment(text)[0])
            
        if 'nlpaug_char' in augmentations and self.char_aug:
            augmented_texts.append(self.char_aug.augment(text)[0])
            
        return augmented_texts

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

def process_chunk(chunk_data, augmenter, augmentations):
    """Process a chunk of data for parallel processing."""
    augmented_texts = []
    augmented_labels = []
    
    for text, label in chunk_data:
        # Original text
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        # Generate augmented versions
        augmented_versions = augmenter.augment_single_text(text, augmentations)
        augmented_texts.extend(augmented_versions[1:])  # Skip the original text
        augmented_labels.extend([label] * (len(augmented_versions) - 1))
    
    return augmented_texts, augmented_labels

def main():
    # Configurazione delle tecniche di data augmentation
    augmentation_config = {
        'synonym': {
            'enabled': True,
            'n_words': 2
        },
        'deletion': {
            'enabled': True,
            'p': 0.15
        },
        'swap': {
            'enabled': True,
            'n_swaps': 2
        }
    }

    # Load the dataset
    df_train = pd.read_json('tweet_sentiment.train.jsonl', lines=True)
    df_test = pd.read_json('tweet_sentiment.test.jsonl', lines=True)

    # Print dataset information
    print("\nDataset Information:")
    print("-" * 40)
    print("Training set shape:", df_train.shape)
    print("Test set shape:", df_test.shape)
    
    # Initialize the augmenter
    augmenter = DataAugmenter(use_advanced_augmentations=False)
    
    # Get enabled augmentations
    enabled_augmentations = [k for k, v in augmentation_config.items() if v['enabled']]
    print("\nEnabled augmentation techniques:", enabled_augmentations)
    
    # Initialize timing
    total_start_time = time.time()
    
    # Create augmented dataset
    augmented_texts = []
    augmented_labels = []
    
    print("\nPerforming data augmentation...")
    # Process tweets with progress bar
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Augmenting tweets"):
        # Original text
        augmented_texts.append(row['text'])
        augmented_labels.append(row['label'])
        
        # Generate augmented versions
        augmented_versions = augmenter.augment_single_text(row['text'], enabled_augmentations)
        augmented_texts.extend(augmented_versions[1:])  # Skip the original text
        augmented_labels.extend([row['label']] * (len(augmented_versions) - 1))
    
    total_time = time.time() - total_start_time
    
    # Create new augmented dataframe
    df_train_augmented = pd.DataFrame({
        'text': augmented_texts,
        'label': augmented_labels
    })
    
    print("\nData Augmentation Statistics:")
    print("-" * 40)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Original training set size: {len(df_train)}")
    print(f"Augmented training set size: {len(df_train_augmented)}")
    print(f"Average time per tweet: {total_time/len(df_train):.2f} seconds")
    
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
    
    # Analyze augmented dataset distribution
    aug_counts, aug_percentages = analyze_class_distribution(df_train_augmented, "Augmented Training Set")
    
    # Load GloVe embeddings
    print("\nLoading GloVe embeddings...")
    glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')
    print(f"Number of words in vocabulary: {len(glove_embeddings)}")
    print(f"Vector dimension: {len(next(iter(glove_embeddings.values())))}")
    
    # Convert tweets to vectors
    print("\nConverting tweets to vectors...")
    X_train = np.array([tweet_to_vector(tweet, glove_embeddings, augmenter.stop_words) 
                       for tweet in df_train_augmented['text']])
    y_train = df_train_augmented['label']
    
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
    
    # Calculate class weights
    class_counts = np.bincount(y_train_numeric)
    total_samples = len(y_train_numeric)
    class_weights = total_samples / (len(np.unique(y_train_numeric)) * class_counts)
    class_weights = class_weights / np.sum(class_weights)
    
    # Train XGBoost model with optimized parameters
    print("\nTraining XGBoost model with optimized parameters...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=len(np.unique(y_train_numeric)),
        scale_pos_weight=class_weights,
        eval_metric=['mlogloss', 'merror'],
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1.0
    )
    
    # Fit the model
    xgb_model.fit(
        X_train_scaled, 
        y_train_numeric,
        eval_set=[(X_val_scaled, y_val_numeric)],
        verbose=True
    )
    
    # Make predictions
    y_train_pred = xgb_model.predict(X_train_scaled)
    y_val_pred = xgb_model.predict(X_val_scaled)
    
    # Calculate metrics
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
    print("\nDetailed Classification Report:")
    print(classification_report(y_train_numeric, y_train_pred))
    
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
    print("\nDetailed Classification Report:")
    print(classification_report(y_val_numeric, y_val_pred))
    
    # Test set evaluation
    print("\nEvaluating on test set...")
    X_test = np.array([tweet_to_vector(tweet, glove_embeddings, augmenter.stop_words) 
                      for tweet in df_test['text']])
    X_test_scaled = scaler.transform(X_test)
    y_test = df_test['label']
    y_test_numeric = label_encoder.transform(y_test)
    y_test_pred = xgb_model.predict(X_test_scaled)
    
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
    print(classification_report(y_test_numeric, y_test_pred))
    
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

    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgb_model, max_num_features=20)
    plt.title('Feature Importance')
    plt.show()

    # Learning Curves
    plt.figure(figsize=(10, 6))
    plt.plot(xgb_model.evals_result()['validation_0']['mlogloss'], label='Validation Loss')
    plt.title('XGBoost Learning Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Multi-class Log Loss')
    plt.legend()
    plt.show()
    
    # Save the model and components
    joblib.dump(xgb_model, 'sentiment_model_xgb_augmented_balanced.joblib')
    joblib.dump(scaler, 'scaler_xgb_augmented_balanced.joblib')
    joblib.dump(label_encoder, 'label_encoder_xgb_augmented_balanced.joblib')
    print("\nModel, scaler and label encoder saved")

if __name__ == "__main__":
    main() 