import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score
from collections import Counter
import re
import os
import urllib.request
import zipfile

from .model import SpamClassifier


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "data_path": "data/text_spam_classification.csv",
    "test_size": 0.30,           # 70/30 split
    "random_state": 42,
    "max_vocab_size": 10000,
    "max_seq_length": 100,
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "n_folds": 3,                # For cross-validation
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # GloVe settings
    "glove_dim": 100,            # GloVe embedding dimension (50, 100, 200, or 300)
    "glove_dir": "data/glove",   # Directory to store GloVe files
    "freeze_embeddings": False,  # Whether to freeze pre-trained embeddings
    # Risk thresholds for 3-tier classification
    # 0 = Not spam (prob < medium_threshold)
    # 1 = Medium risk (medium_threshold <= prob < high_threshold)  
    # 2 = High risk (prob >= high_threshold)
    "medium_threshold": 0.4,
    "high_threshold": 0.7,
    # Early stopping
    "early_stopping_patience": 1,  # Stop if validation loss doesn't improve for N epochs
}


# ============================================================================
# GloVe Embeddings
# ============================================================================

def download_glove(glove_dir: str, dim: int = 100):
    """Download GloVe Twitter embeddings if not already present."""
    os.makedirs(glove_dir, exist_ok=True)
    
    glove_file = os.path.join(glove_dir, f"glove.twitter.27B.{dim}d.txt")
    
    if os.path.exists(glove_file):
        print(f"GloVe file already exists: {glove_file}")
        return glove_file
    
    # Download GloVe Twitter embeddings
    zip_path = os.path.join(glove_dir, "glove.twitter.27B.zip")
    url = "https://nlp.stanford.edu/data/glove.twitter.27B.zip"
    
    if not os.path.exists(zip_path):
        print(f"Downloading GloVe Twitter embeddings (~1.4GB)...")
        print(f"URL: {url}")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
    
    # Extract
    print("Extracting GloVe files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(glove_dir)
    print("Extraction complete!")
    
    return glove_file


def load_glove_embeddings(glove_path: str) -> dict:
    """Load GloVe embeddings from file into a dictionary."""
    print(f"Loading GloVe embeddings from {glove_path}...")
    
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
    
    print(f"Loaded {len(embeddings):,} word vectors")
    return embeddings


def build_embedding_matrix(word2idx: dict, glove_embeddings: dict, embedding_dim: int) -> np.ndarray:
    """
    Build embedding matrix for our vocabulary using GloVe vectors.
    Words not in GloVe get random initialization.
    """
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    
    found = 0
    not_found = 0
    
    for word, idx in word2idx.items():
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]
            found += 1
        else:
            # Random initialization for unknown words (except PAD which stays zero)
            if idx != 0:  # Skip PAD token
                embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)
            not_found += 1
    
    coverage = found / vocab_size * 100
    print(f"Embedding coverage: {found}/{vocab_size} ({coverage:.1f}%) words found in GloVe")
    print(f"Words not in GloVe: {not_found} (randomly initialized)")
    
    return embedding_matrix


def get_or_build_embedding_matrix(tokenizer, config) -> np.ndarray:
    """
    Load cached embedding matrix if available, otherwise build and cache it.
    This avoids reloading the full GloVe file on every run.
    """
    cache_dir = config["glove_dir"]
    cache_file = os.path.join(cache_dir, f"embedding_matrix_{config['glove_dim']}d_{len(tokenizer.word2idx)}.npy")
    vocab_cache = os.path.join(cache_dir, f"vocab_hash_{config['glove_dim']}d.npy")
    
    # Create a simple hash of vocabulary to detect changes
    vocab_hash = hash(frozenset(tokenizer.word2idx.items()))
    
    # Check if cached matrix exists and vocab hasn't changed
    if os.path.exists(cache_file) and os.path.exists(vocab_cache):
        cached_hash = np.load(vocab_cache, allow_pickle=True).item()
        if cached_hash == vocab_hash:
            print(f"Loading cached embedding matrix from {cache_file}")
            return np.load(cache_file)
        else:
            print("Vocabulary changed, rebuilding embedding matrix...")
    
    # Need to build fresh - load GloVe
    glove_path = download_glove(config["glove_dir"], config["glove_dim"])
    glove_embeddings = load_glove_embeddings(glove_path)
    
    # Build embedding matrix
    embedding_matrix = build_embedding_matrix(
        tokenizer.word2idx, 
        glove_embeddings, 
        config["glove_dim"]
    )
    
    # Cache for next time
    np.save(cache_file, embedding_matrix)
    np.save(vocab_cache, vocab_hash)
    print(f"Cached embedding matrix to {cache_file}")
    
    # Free memory
    del glove_embeddings
    
    return embedding_matrix


# ============================================================================
# Text Preprocessing & Tokenization
# ============================================================================

class SimpleTokenizer:
    """Simple word-level tokenizer with vocabulary."""
    
    def __init__(self, max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
    
    def fit(self, texts: list[str]):
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)
        
        # Keep most common words
        most_common = word_counts.most_common(self.max_vocab_size - 2)  # -2 for PAD, UNK
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx)}")
    
    def encode(self, text: str, max_length: int) -> list[int]:
        """Convert text to list of token indices."""
        tokens = self._tokenize(text)
        indices = [self.word2idx.get(t, 1) for t in tokens]  # 1 = UNK
        
        # Pad or truncate
        if len(indices) < max_length:
            indices = indices + [0] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        
        return indices
    
    def _tokenize(self, text: str) -> list[str]:
        """Basic tokenization: lowercase, remove special chars, split."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()


# ============================================================================
# Dataset
# ============================================================================

class SpamDataset(Dataset):
    """PyTorch Dataset for spam classification."""
    
    def __init__(self, texts: list[str], labels: list[int], tokenizer: SimpleTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoded = self.tokenizer.encode(text, self.max_length)
        
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float)
        }


# ============================================================================
# Training & Evaluation
# ============================================================================

def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Compute class weight for imbalanced data (inverse frequency)."""
    counts = Counter(labels)
    total = len(labels)
    
    # Weight = total / (num_classes * count)
    weight_0 = total / (2 * counts[0])
    weight_1 = total / (2 * counts[1])
    
    print(f"Class distribution: {counts}")
    print(f"Class weights: 0={weight_0:.3f}, 1={weight_1:.3f}")
    
    # Return weight for positive class (used in BCEWithLogitsLoss)
    return torch.tensor(weight_1 / weight_0)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, medium_threshold=0.4, high_threshold=0.7):
    """
    Evaluate model and return loss + metrics.
    
    Uses 3-tier risk classification:
        0 = Not spam (prob < medium_threshold)
        1 = Medium risk (medium_threshold <= prob < high_threshold)
        2 = High risk (prob >= high_threshold)
    
    For binary metrics, maps: 0 -> not spam, 1 or 2 -> spam
    """
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids).squeeze(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 3-tier risk predictions
    risk_preds = np.zeros_like(all_probs, dtype=int)
    risk_preds[all_probs >= medium_threshold] = 1  # Medium risk
    risk_preds[all_probs >= high_threshold] = 2     # High risk
    
    # Count distribution of risk levels
    risk_counts = {
        0: np.sum(risk_preds == 0),
        1: np.sum(risk_preds == 1),
        2: np.sum(risk_preds == 2)
    }
    
    # Map to binary for metrics: 0 = not spam, 1 or 2 = spam
    binary_preds = (risk_preds >= 1).astype(int)
    
    # Compute binary metrics
    accuracy = np.mean(binary_preds == all_labels)
    
    tp = np.sum((binary_preds == 1) & (all_labels == 1))
    fp = np.sum((binary_preds == 1) & (all_labels == 0))
    fn = np.sum((binary_preds == 0) & (all_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Compute PR-AUC (Area Under Precision-Recall Curve)
    pr_auc = average_precision_score(all_labels, all_probs)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "risk_distribution": risk_counts,
        "risk_preds": risk_preds,
    }


# ============================================================================
# Cross-Validation for Hyperparameter Tuning
# ============================================================================

def cross_validate(texts, labels, tokenizer, embedding_matrix, hyperparams, config):
    """Run K-Fold cross-validation and return average F1 score."""
    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=config["random_state"])
    fold_scores = []
    
    print(f"\n--- Cross-validating: hidden_dim={hyperparams['hidden_dim']}, dropout={hyperparams['dropout']} ---")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        train_dataset = SpamDataset(train_texts, train_labels, tokenizer, config["max_seq_length"])
        val_dataset = SpamDataset(val_texts, val_labels, tokenizer, config["max_seq_length"])
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
        
        # Create model with pre-trained embeddings
        model = SpamClassifier(
            vocab_size=len(tokenizer.word2idx),
            embedding_dim=config["glove_dim"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            pretrained_embeddings=embedding_matrix,
            freeze_embeddings=config["freeze_embeddings"]
        ).to(config["device"])
        
        # Weighted loss for class imbalance
        pos_weight = compute_class_weights(train_labels).to(config["device"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
        
        # Train for fewer epochs during CV
        cv_epochs = 3
        for epoch in range(cv_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, config["device"])
        
        # Evaluate with 3-tier risk classification
        metrics = evaluate(
            model, val_loader, criterion, config["device"],
            medium_threshold=config["medium_threshold"],
            high_threshold=config["high_threshold"]
        )
        fold_scores.append(metrics["f1"])
        risk_dist = metrics["risk_distribution"]
        print(f"  Fold {fold + 1}: F1 = {metrics['f1']:.4f} | Risk dist: 0={risk_dist[0]}, 1={risk_dist[1]}, 2={risk_dist[2]}")
    
    avg_f1 = np.mean(fold_scores)
    print(f"  Average F1: {avg_f1:.4f}")
    return avg_f1


def hyperparameter_search(train_texts, train_labels, tokenizer, embedding_matrix, config):
    """Simple grid search over hyperparameters using cross-validation."""
    
    # Define search space (embedding_dim is fixed by GloVe)
    param_grid = [
        {"hidden_dim": 128, "dropout": 0.3, "lr": 1e-3},
        {"hidden_dim": 256, "dropout": 0.3, "lr": 1e-3},
        {"hidden_dim": 256, "dropout": 0.5, "lr": 1e-3},
    ]
    
    best_score = 0
    best_params = None
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH (with GloVe embeddings)")
    print("=" * 60)
    
    for params in param_grid:
        score = cross_validate(train_texts, train_labels, tokenizer, embedding_matrix, params, config)
        if score > best_score:
            best_score = score
            best_params = params
    
    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best CV F1 score: {best_score:.4f}")
    
    return best_params


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    print("=" * 60)
    print("SPAM CLASSIFIER TRAINING (with GloVe Embeddings)")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"GloVe dimension: {CONFIG['glove_dim']}")
    
    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv(CONFIG["data_path"])
    texts = df["sms"].astype(str).tolist()
    labels = df["label"].tolist()
    print(f"Total samples: {len(texts)}")
    
    # Split data 70/30
    print("\n[2] Splitting data (70/30)...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=labels  # Maintain class distribution
    )
    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Build tokenizer on training data only
    print("\n[3] Building vocabulary...")
    tokenizer = SimpleTokenizer(max_vocab_size=CONFIG["max_vocab_size"])
    tokenizer.fit(train_texts)
    
    # Load or build embedding matrix (cached for speed)
    print("\n[4] Loading embeddings...")
    embedding_matrix = get_or_build_embedding_matrix(tokenizer, CONFIG)
    
    # Hyperparameter search with cross-validation
    print("\n[5] Hyperparameter tuning with cross-validation...")
    best_params = hyperparameter_search(train_texts, train_labels, tokenizer, embedding_matrix, CONFIG)
    
    # Train final model with best hyperparameters on full training set
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)
    
    train_dataset = SpamDataset(train_texts, train_labels, tokenizer, CONFIG["max_seq_length"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    # TODO: Uncomment when ready to evaluate on test set
    # test_dataset = SpamDataset(test_texts, test_labels, tokenizer, CONFIG["max_seq_length"])
    # test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])
    
    # Create model with best params and pre-trained embeddings
    model = SpamClassifier(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=CONFIG["glove_dim"],
        hidden_dim=best_params["hidden_dim"],
        dropout=best_params["dropout"],
        pretrained_embeddings=embedding_matrix,
        freeze_embeddings=CONFIG["freeze_embeddings"]
    ).to(CONFIG["device"])
    
    # Weighted loss for class imbalance
    pos_weight = compute_class_weights(train_labels).to(CONFIG["device"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    
    # Training loop with early stopping
    print("\nTraining (with early stopping)...")
    best_train_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG["device"])
        
        # Evaluate on training set
        train_metrics = evaluate(
            model, train_loader, criterion, CONFIG["device"],
            medium_threshold=CONFIG["medium_threshold"],
            high_threshold=CONFIG["high_threshold"]
        )
        
        # Print training metrics
        risk_dist = train_metrics["risk_distribution"]
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']} | "
              f"Loss: {train_loss:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"Prec: {train_metrics['precision']:.4f} | "
              f"Rec: {train_metrics['recall']:.4f} | "
              f"F1: {train_metrics['f1']:.4f} | "
              f"PR-AUC: {train_metrics['pr_auc']:.4f} | "
              f"Risk: [{risk_dist[0]}/{risk_dist[1]}/{risk_dist[2]}]")
        
        # Early stopping check (based on training loss since test is commented out)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            print(f"  -> No improvement in loss (patience: {patience_counter}/{CONFIG['early_stopping_patience']})")
            
            if patience_counter >= CONFIG["early_stopping_patience"]:
                print(f"  -> Early stopping triggered at epoch {epoch + 1}")
                # Restore best model
                model.load_state_dict(best_model_state)
                break
    
    # Final training metrics
    print("\n" + "=" * 60)
    print("FINAL TRAINING METRICS")
    print("=" * 60)
    final_train_metrics = evaluate(
        model, train_loader, criterion, CONFIG["device"],
        medium_threshold=CONFIG["medium_threshold"],
        high_threshold=CONFIG["high_threshold"]
    )
    print(f"Accuracy:  {final_train_metrics['accuracy']:.4f}")
    print(f"Precision: {final_train_metrics['precision']:.4f}")
    print(f"Recall:    {final_train_metrics['recall']:.4f}")
    print(f"F1 Score:  {final_train_metrics['f1']:.4f}")
    print(f"PR-AUC:    {final_train_metrics['pr_auc']:.4f}")
    
    risk_dist = final_train_metrics["risk_distribution"]
    total = sum(risk_dist.values())
    print(f"\nRisk Distribution on Training Set:")
    print(f"  Not Spam (0):    {risk_dist[0]:4d} ({risk_dist[0]/total*100:.1f}%)")
    print(f"  Medium Risk (1): {risk_dist[1]:4d} ({risk_dist[1]/total*100:.1f}%)")
    print(f"  High Risk (2):   {risk_dist[2]:4d} ({risk_dist[2]/total*100:.1f}%)")
    
    # TODO: Uncomment when ready to evaluate on test set
    # print("\n" + "=" * 60)
    # print("FINAL EVALUATION ON TEST SET")
    # print("=" * 60)
    # final_metrics = evaluate(
    #     model, test_loader, criterion, CONFIG["device"],
    #     medium_threshold=CONFIG["medium_threshold"],
    #     high_threshold=CONFIG["high_threshold"]
    # )
    # print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    # print(f"Precision: {final_metrics['precision']:.4f}")
    # print(f"Recall:    {final_metrics['recall']:.4f}")
    # print(f"F1 Score:  {final_metrics['f1']:.4f}")
    # print(f"PR-AUC:    {final_metrics['pr_auc']:.4f}")
    # risk_dist = final_metrics["risk_distribution"]
    # total = sum(risk_dist.values())
    # print(f"\nRisk Distribution on Test Set:")
    # print(f"  Not Spam (0):    {risk_dist[0]:4d} ({risk_dist[0]/total*100:.1f}%)")
    # print(f"  Medium Risk (1): {risk_dist[1]:4d} ({risk_dist[1]/total*100:.1f}%)")
    # print(f"  High Risk (2):   {risk_dist[2]:4d} ({risk_dist[2]/total*100:.1f}%)")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer_word2idx": tokenizer.word2idx,
        "config": CONFIG,
        "best_params": best_params,
        "embedding_dim": CONFIG["glove_dim"]
    }, "models/spam_classifier.pt")
    print("\nModel saved to models/spam_classifier.pt")


if __name__ == "__main__":
    main()
