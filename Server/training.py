import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter
import re
import os

from model import SpamClassifier


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
}


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


def evaluate(model, dataloader, criterion, device):
    """Evaluate model and return loss + metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids).squeeze(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Compute precision, recall, F1 for spam class
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================================
# Cross-Validation for Hyperparameter Tuning
# ============================================================================

def cross_validate(texts, labels, tokenizer, hyperparams, config):
    """Run K-Fold cross-validation and return average F1 score."""
    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=config["random_state"])
    fold_scores = []
    
    print(f"\n--- Cross-validating: {hyperparams} ---")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        train_dataset = SpamDataset(train_texts, train_labels, tokenizer, config["max_seq_length"])
        val_dataset = SpamDataset(val_texts, val_labels, tokenizer, config["max_seq_length"])
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
        
        # Create model with hyperparams
        model = SpamClassifier(
            vocab_size=len(tokenizer.word2idx),
            embedding_dim=hyperparams["embedding_dim"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"]
        ).to(config["device"])
        
        # Weighted loss for class imbalance
        pos_weight = compute_class_weights(train_labels).to(config["device"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
        
        # Train for fewer epochs during CV
        cv_epochs = 3
        for epoch in range(cv_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, config["device"])
        
        # Evaluate
        metrics = evaluate(model, val_loader, criterion, config["device"])
        fold_scores.append(metrics["f1"])
        print(f"  Fold {fold + 1}: F1 = {metrics['f1']:.4f}")
    
    avg_f1 = np.mean(fold_scores)
    print(f"  Average F1: {avg_f1:.4f}")
    return avg_f1


def hyperparameter_search(train_texts, train_labels, tokenizer, config):
    """Simple grid search over hyperparameters using cross-validation."""
    
    # Define search space (keeping it simple)
    param_grid = [
        {"embedding_dim": 64, "hidden_dim": 128, "dropout": 0.3, "lr": 1e-3},
        {"embedding_dim": 128, "hidden_dim": 256, "dropout": 0.3, "lr": 1e-3},
        {"embedding_dim": 128, "hidden_dim": 256, "dropout": 0.5, "lr": 1e-3},
    ]
    
    best_score = 0
    best_params = None
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH")
    print("=" * 60)
    
    for params in param_grid:
        score = cross_validate(train_texts, train_labels, tokenizer, params, config)
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
    print("SPAM CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    
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
    
    # Hyperparameter search with cross-validation
    print("\n[4] Hyperparameter tuning with cross-validation...")
    best_params = hyperparameter_search(train_texts, train_labels, tokenizer, CONFIG)
    
    # Train final model with best hyperparameters on full training set
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)
    
    train_dataset = SpamDataset(train_texts, train_labels, tokenizer, CONFIG["max_seq_length"])
    test_dataset = SpamDataset(test_texts, test_labels, tokenizer, CONFIG["max_seq_length"])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])
    
    # Create model with best params
    model = SpamClassifier(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=best_params["embedding_dim"],
        hidden_dim=best_params["hidden_dim"],
        dropout=best_params["dropout"]
    ).to(CONFIG["device"])
    
    # Weighted loss for class imbalance
    pos_weight = compute_class_weights(train_labels).to(CONFIG["device"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    
    # Training loop
    print("\nTraining...")
    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG["device"])
        val_metrics = evaluate(model, test_loader, criterion, CONFIG["device"])
        
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    final_metrics = evaluate(model, test_loader, criterion, CONFIG["device"])
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1 Score:  {final_metrics['f1']:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer_word2idx": tokenizer.word2idx,
        "config": CONFIG,
        "best_params": best_params
    }, "models/spam_classifier.pt")
    print("\nModel saved to models/spam_classifier.pt")


if __name__ == "__main__":
    main()
