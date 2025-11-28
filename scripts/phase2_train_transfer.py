"""
Phase 2: Transfer Learning LSTM Training
=========================================
Two-stage training for de novo drug generation:

Stage 1 - PRE-TRAINING (General Brain):
    Train on 50k general ChEMBL molecules to learn chemistry grammar
    
Stage 2 - FINE-TUNING (Specialist Brain):
    Load pre-trained model, fine-tune on EGFR-specific molecules

Usage:
    # Stage 1: Pre-train on general molecules
    python scripts/phase2_train_transfer.py --mode pretrain --epochs 50
    
    # Stage 2: Fine-tune on EGFR molecules
    python scripts/phase2_train_transfer.py --mode finetune --epochs 30
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================================
# SMILES TOKENIZER
# ============================================================================

SMILES_REGEX = re.compile(r"""
    (\[[^\]]+\]|       # Bracketed atoms like [nH], [O-], [C@@H]
    Br|Cl|              # Two-letter elements
    \#|                 # Triple bond
    [BCNOPSFIbcnops]|  # Single-letter organic atoms
    [0-9]|              # Ring numbers
    \(|\)|             # Parentheses  
    =|                  # Double bond
    -|                  # Single bond (explicit)
    \+|                 # Positive charge
    \\|/|              # Stereochemistry
    @|                  # Chirality
    \.|                 # Disconnected structures
    .)                  # Any other character (fallback)
""", re.VERBOSE)


def tokenize_smiles(smiles):
    """
    Tokenize SMILES into chemically meaningful tokens.
    Properly handles: Cl, Br, [nH], [O-], [C@@H], etc.
    """
    return SMILES_REGEX.findall(smiles)


def detokenize_smiles(tokens):
    """Convert token list back to SMILES string."""
    return ''.join(tokens)


# ============================================================================
# DATASET
# ============================================================================

class SMILESDataset(Dataset):
    """Dataset for SMILES with proper tokenization."""
    
    def __init__(self, smiles_list, token_to_idx, seq_length=100):
        self.smiles_list = smiles_list
        self.token_to_idx = token_to_idx
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        tokens = tokenize_smiles(smiles)
        tokens = ['<START>'] + tokens + ['<END>']
        
        # Convert to indices
        indices = [self.token_to_idx.get(t, self.token_to_idx['<UNK>']) for t in tokens]
        
        # Pad or truncate
        if len(indices) < self.seq_length:
            indices = indices + [self.token_to_idx['<PAD>']] * (self.seq_length - len(indices))
        else:
            indices = indices[:self.seq_length]
        
        input_seq = torch.tensor(indices[:-1], dtype=torch.long)
        target_seq = torch.tensor(indices[1:], dtype=torch.long)
        
        return input_seq, target_seq


# ============================================================================
# LSTM MODEL
# ============================================================================

class LSTMModel(nn.Module):
    """Token-level LSTM for SMILES generation."""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, 
                 num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


# ============================================================================
# TRAINER
# ============================================================================

class TransferTrainer:
    """Training manager with transfer learning support."""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=0.001, save_dir='models', mode='pretrain'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.mode = mode
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'checkpoints').mkdir(exist_ok=True)
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (input_seq, target_seq) in enumerate(self.train_loader):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            
            self.optimizer.zero_grad()
            output, _ = self.model(input_seq)
            
            loss = self.criterion(
                output.reshape(-1, output.size(-1)),
                target_seq.reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f'  Batch [{batch_idx + 1}/{len(self.train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for input_seq, target_seq in self.val_loader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                output, _ = self.model(input_seq)
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    target_seq.reshape(-1)
                )
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs, save_every=10):
        """Train model for multiple epochs."""
        print(f"\n{'='*60}")
        print(f"STARTING {self.mode.upper()} TRAINING")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            
            # Loss interpretation
            if train_loss < 0.5:
                print(f"   ✅ Model understands chemistry grammar!")
            elif train_loss < 1.0:
                print(f"   📈 Good progress, learning patterns...")
            else:
                print(f"   🔄 Still learning basic patterns...")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_name = f'{self.mode}_best_model.pth'
                self.save_model(model_name)
                print(f"   ✓ Best model saved: {model_name}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self._save_checkpoint(epoch, train_loss, val_loss)
        
        # Save final model
        final_name = f'{self.mode}_final_model.pth'
        self.save_model(final_name)
        print(f"\n✓ Training complete! Final model: {final_name}")
        
        # Plot curves
        self.plot_training_curves()
        
        return self.train_losses, self.val_losses
    
    def save_model(self, filename):
        """Save model state."""
        filepath = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'mode': self.mode,
            'vocab_size': self.model.vocab_size,
        }, filepath)
    
    def _save_checkpoint(self, epoch, train_loss, val_loss):
        """Save training checkpoint."""
        checkpoint_path = self.save_dir / 'checkpoints' / f'{self.mode}_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"   ✓ Checkpoint: {self.mode}_epoch_{epoch}.pth")
    
    def plot_training_curves(self):
        """Plot training and validation loss."""
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'orange', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{self.mode.upper()} Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add target line at 0.5
        plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Target (grammar learned)')
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = PROJECT_ROOT / 'results' / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / f'{self.mode}_training_loss.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training curve saved: {plot_path}")
        plt.close()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_smiles(filepath):
    """Load SMILES from file."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def build_vocabulary(smiles_list):
    """Build token vocabulary from SMILES."""
    all_tokens = set()
    for smiles in smiles_list:
        tokens = tokenize_smiles(smiles)
        all_tokens.update(tokens)
    
    tokens_sorted = sorted(list(all_tokens))
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    vocab = special_tokens + tokens_sorted
    
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_token = {idx: token for idx, token in enumerate(vocab)}
    
    return token_to_idx, idx_to_token, vocab


def save_vocabulary(token_to_idx, idx_to_token, vocab, save_path, mode='pretrain'):
    """Save vocabulary to JSON file."""
    vocab_data = {
        'token_to_idx': token_to_idx,
        'idx_to_token': {int(k): v for k, v in idx_to_token.items()},
        'vocab': vocab,
        'vocab_size': len(vocab),
        'tokenizer': 'regex',
        'mode': mode,
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)


def load_vocabulary(vocab_path):
    """Load vocabulary from JSON file."""
    with open(vocab_path, 'r') as f:
        data = json.load(f)
    
    token_to_idx = data['token_to_idx']
    idx_to_token = {int(k): v for k, v in data['idx_to_token'].items()}
    vocab = data['vocab']
    
    return token_to_idx, idx_to_token, vocab


def analyze_dataset(smiles_list, name="Dataset"):
    """Analyze SMILES dataset."""
    print(f"\n📊 {name} Analysis:")
    print(f"   Total SMILES: {len(smiles_list):,}")
    
    token_lengths = [len(tokenize_smiles(s)) for s in smiles_list]
    print(f"   Token length: {np.mean(token_lengths):.1f} avg, {max(token_lengths)} max")
    
    all_tokens = set()
    for smiles in smiles_list:
        all_tokens.update(tokenize_smiles(smiles))
    print(f"   Unique tokens: {len(all_tokens)}")
    
    # Multi-char tokens
    multi_char = [t for t in all_tokens if len(t) > 1]
    print(f"   Multi-char tokens: {len(multi_char)} (Cl, Br, [nH], etc.)")


# ============================================================================
# MAIN: PRE-TRAINING MODE
# ============================================================================

def run_pretrain(args):
    """Run pre-training on general ChEMBL data."""
    print("\n" + "="*60)
    print("🧠 ACTION 1: PRE-TRAINING (General Chemistry Brain)")
    print("="*60)
    print("\nGoal: Teach the model how to write valid SMILES")
    print("      (brackets, rings, bonds, elements)")
    print("\nTarget: Loss < 0.5 means grammar is learned!")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Load general data
    data_path = DATA_DIR / 'general_chembl.txt'
    if not data_path.exists():
        print(f"\n❌ Error: {data_path} not found!")
        print("   Run: python scripts/extract_general_chembl.py")
        return
    
    smiles_list = load_smiles(data_path)
    print(f"\n✓ Loaded {len(smiles_list):,} general SMILES")
    
    analyze_dataset(smiles_list, "General ChEMBL")
    
    # Build vocabulary
    print(f"\n{'='*60}")
    print("BUILDING VOCABULARY")
    print(f"{'='*60}")
    token_to_idx, idx_to_token, vocab = build_vocabulary(smiles_list)
    print(f"✓ Vocabulary size: {len(vocab)} tokens")
    
    # Save vocabulary
    vocab_path = MODELS_DIR / 'pretrain_vocab.json'
    save_vocabulary(token_to_idx, idx_to_token, vocab, vocab_path, mode='pretrain')
    print(f"✓ Vocabulary saved: {vocab_path}")
    
    # Split data
    np.random.seed(args.seed)
    np.random.shuffle(smiles_list)
    split_idx = int(len(smiles_list) * (1 - args.val_split))
    train_smiles = smiles_list[:split_idx]
    val_smiles = smiles_list[split_idx:]
    
    print(f"\n📂 Data split:")
    print(f"   Training:   {len(train_smiles):,}")
    print(f"   Validation: {len(val_smiles):,}")
    
    # Create datasets
    train_dataset = SMILESDataset(train_smiles, token_to_idx, args.seq_length)
    val_dataset = SMILESDataset(val_smiles, token_to_idx, args.seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    # Create model
    print(f"\n{'='*60}")
    print("BUILDING MODEL")
    print(f"{'='*60}")
    model = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")
    
    # Train
    trainer = TransferTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        save_dir=str(MODELS_DIR),
        mode='pretrain'
    )
    
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)
    
    print("\n" + "="*60)
    print("✅ PRE-TRAINING COMPLETE!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - models/pretrain_best_model.pth")
    print(f"  - models/pretrain_final_model.pth")
    print(f"  - models/pretrain_vocab.json")
    print(f"  - results/plots/pretrain_training_loss.png")
    print(f"\nNext step: Fine-tune on EGFR data")
    print(f"  python scripts/phase2_train_transfer.py --mode finetune --epochs 30")


# ============================================================================
# MAIN: FINE-TUNING MODE
# ============================================================================

def run_finetune(args):
    """Run fine-tuning on EGFR-specific data."""
    print("\n" + "="*60)
    print("🎯 ACTION 2: FINE-TUNING (EGFR Specialist Brain)")
    print("="*60)
    print("\nGoal: Specialize the model on EGFR inhibitor patterns")
    print("      (Force it to 'forget' irrelevant molecules)")
    print("\nExpected: Loss may spike briefly, then settle down")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Check for pre-trained model
    pretrain_model_path = MODELS_DIR / 'pretrain_best_model.pth'
    pretrain_vocab_path = MODELS_DIR / 'pretrain_vocab.json'
    
    if not pretrain_model_path.exists():
        print(f"\n❌ Error: Pre-trained model not found!")
        print(f"   Expected: {pretrain_model_path}")
        print(f"   Run pre-training first: --mode pretrain")
        return
    
    # Load EGFR data
    egfr_path = DATA_DIR / 'clean_smiles.txt'
    if not egfr_path.exists():
        print(f"\n❌ Error: {egfr_path} not found!")
        return
    
    egfr_smiles = load_smiles(egfr_path)
    print(f"\n✓ Loaded {len(egfr_smiles):,} EGFR inhibitor SMILES")
    
    analyze_dataset(egfr_smiles, "EGFR Inhibitors")
    
    # Load pre-trained vocabulary
    print(f"\n{'='*60}")
    print("LOADING PRE-TRAINED MODEL")
    print(f"{'='*60}")
    
    token_to_idx, idx_to_token, vocab = load_vocabulary(pretrain_vocab_path)
    print(f"✓ Vocabulary: {len(vocab)} tokens (from pre-training)")
    
    # Check if EGFR data has new tokens
    egfr_tokens = set()
    for smiles in egfr_smiles:
        egfr_tokens.update(tokenize_smiles(smiles))
    
    new_tokens = egfr_tokens - set(vocab)
    if new_tokens:
        print(f"⚠️  {len(new_tokens)} new tokens in EGFR data (will use <UNK>)")
        print(f"   Examples: {list(new_tokens)[:5]}")
    else:
        print(f"✓ All EGFR tokens covered by pre-trained vocabulary")
    
    # Load model
    checkpoint = torch.load(pretrain_model_path, map_location=device)
    
    model = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded pre-trained weights")
    print(f"   Pre-train best loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    # Split EGFR data
    np.random.seed(args.seed)
    np.random.shuffle(egfr_smiles)
    split_idx = int(len(egfr_smiles) * (1 - args.val_split))
    train_smiles = egfr_smiles[:split_idx]
    val_smiles = egfr_smiles[split_idx:]
    
    print(f"\n📂 EGFR data split:")
    print(f"   Training:   {len(train_smiles):,}")
    print(f"   Validation: {len(val_smiles):,}")
    
    # Create datasets
    train_dataset = SMILESDataset(train_smiles, token_to_idx, args.seq_length)
    val_dataset = SMILESDataset(val_smiles, token_to_idx, args.seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    # Fine-tune (use lower learning rate)
    finetune_lr = args.learning_rate * 0.5  # Lower LR for fine-tuning
    print(f"\n📉 Fine-tuning learning rate: {finetune_lr} (50% of pre-train)")
    
    trainer = TransferTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=finetune_lr,
        save_dir=str(MODELS_DIR),
        mode='finetune'
    )
    
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)
    
    print("\n" + "="*60)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - models/finetune_best_model.pth (EGFR specialist)")
    print(f"  - models/finetune_final_model.pth")
    print(f"  - results/plots/finetune_training_loss.png")
    print(f"\n🎯 The final model is now an EGFR inhibitor expert!")
    print(f"\nNext step: Generate molecules")
    print(f"  python scripts/phase3_generate.py --model finetune_best_model.pth")


# ============================================================================
# COMBINED LOSS PLOT
# ============================================================================

def plot_combined_curves():
    """Plot combined pre-train and fine-tune loss curves."""
    print("\n" + "="*60)
    print("📊 ACTION 3: VERIFYING LOSS CURVES")
    print("="*60)
    
    pretrain_path = MODELS_DIR / 'pretrain_final_model.pth'
    finetune_path = MODELS_DIR / 'finetune_final_model.pth'
    
    if not pretrain_path.exists() or not finetune_path.exists():
        print("❌ Need both pretrain and finetune models to create combined plot")
        return
    
    # Load training histories
    pretrain_data = torch.load(pretrain_path, map_location='cpu')
    finetune_data = torch.load(finetune_path, map_location='cpu')
    
    pretrain_train = pretrain_data['train_losses']
    pretrain_val = pretrain_data['val_losses']
    finetune_train = finetune_data['train_losses']
    finetune_val = finetune_data['val_losses']
    
    # Create combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pre-training plot
    epochs1 = range(1, len(pretrain_train) + 1)
    ax1.plot(epochs1, pretrain_train, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs1, pretrain_val, color='orange', label='Validation Loss', linewidth=2)
    ax1.axhline(y=0.5, color='g', linestyle='--', alpha=0.7, label='Target (grammar learned)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Stage 1: PRE-TRAINING (General Brain)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Fine-tuning plot
    epochs2 = range(1, len(finetune_train) + 1)
    ax2.plot(epochs2, finetune_train, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs2, finetune_val, color='orange', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Stage 2: FINE-TUNING (EGFR Specialist)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save combined plot
    plots_dir = PROJECT_ROOT / 'results' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / 'transfer_learning_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined plot saved: {plot_path}")
    plt.close()
    
    # Analysis
    print(f"\n📈 Loss Analysis:")
    print(f"\n   PRE-TRAINING:")
    print(f"   - Final train loss: {pretrain_train[-1]:.4f}")
    print(f"   - Final val loss:   {pretrain_val[-1]:.4f}")
    if pretrain_val[-1] < 0.5:
        print(f"   ✅ Model learned chemistry grammar!")
    
    print(f"\n   FINE-TUNING:")
    print(f"   - Final train loss: {finetune_train[-1]:.4f}")
    print(f"   - Final val loss:   {finetune_val[-1]:.4f}")
    
    # Check for overfitting
    if finetune_val[-1] > finetune_val[0] * 1.5 and finetune_train[-1] < finetune_train[0] * 0.5:
        print(f"   ⚠️  Warning: Possible overfitting detected!")
        print(f"      (Validation loss increased while training loss decreased)")
    else:
        print(f"   ✅ Training and validation losses moving together - good!")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Transfer Learning LSTM Training')
    
    parser.add_argument('--mode', type=str, default='pretrain',
                        choices=['pretrain', 'finetune', 'plot'],
                        help='Training mode: pretrain, finetune, or plot')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='Maximum sequence length')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("\n" + "="*60)
    print("  TRANSFER LEARNING FOR DE NOVO DRUG GENERATION")
    print("="*60)
    
    if args.mode == 'pretrain':
        run_pretrain(args)
    elif args.mode == 'finetune':
        run_finetune(args)
    elif args.mode == 'plot':
        plot_combined_curves()


if __name__ == "__main__":
    main()
