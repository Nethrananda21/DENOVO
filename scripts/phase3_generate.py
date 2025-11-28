"""
Phase 3: Molecule Generation
============================
Generate novel EGFR inhibitor candidates using the fine-tuned LSTM model.

This script:
1. Loads the fine-tuned model (EGFR specialist)
2. Seeds it with <START> token
3. Samples next tokens using temperature-controlled probability
4. Validates outputs with RDKit
5. Filters for drug-likeness

Usage:
    python scripts/phase3_generate.py --num_samples 5000 --temperature 0.8
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from collections import Counter
from datetime import datetime

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem
from rdkit import RDLogger
from tqdm import tqdm

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================================
# LSTM MODEL (must match training architecture)
# ============================================================================

class LSTMModel(torch.nn.Module):
    """Token-level LSTM for SMILES generation."""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, 
                 num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
        
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
# MOLECULE GENERATOR
# ============================================================================

class MoleculeGenerator:
    """Generate molecules using trained LSTM model."""
    
    def __init__(self, model_path, vocab_path, device='cuda'):
        """
        Initialize generator.
        
        Args:
            model_path: Path to trained model (.pth)
            vocab_path: Path to vocabulary (.json)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        self.token_to_idx = vocab_data['token_to_idx']
        self.idx_to_token = {int(k): v for k, v in vocab_data['idx_to_token'].items()}
        self.vocab_size = len(self.token_to_idx)
        
        print(f"✓ Loaded vocabulary: {self.vocab_size} tokens")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = LSTMModel(
            vocab_size=self.vocab_size,
            embedding_dim=128,
            hidden_dim=512,
            num_layers=2,
            dropout=0.0  # No dropout during generation
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded model from {model_path}")
        print(f"  Device: {self.device}")
    
    def generate_smiles(self, temperature=0.8, max_length=100):
        """
        Generate a single SMILES string.
        
        Args:
            temperature: Sampling temperature (0.5-1.2)
                - Lower = more conservative/common patterns
                - Higher = more creative/diverse
            max_length: Maximum sequence length
            
        Returns:
            Generated SMILES string or None if invalid
        """
        with torch.no_grad():
            # Start with <START> token
            current_token = torch.tensor([[self.token_to_idx['<START>']]]).to(self.device)
            hidden = None
            
            generated_tokens = []
            
            for _ in range(max_length):
                # Get model prediction
                output, hidden = self.model(current_token, hidden)
                
                # Apply temperature scaling
                logits = output[0, -1, :] / temperature
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=0)
                
                # Sample from distribution
                next_idx = torch.multinomial(probs, 1).item()
                next_token = self.idx_to_token[next_idx]
                
                # Check for end token
                if next_token == '<END>':
                    break
                
                # Skip special tokens
                if next_token in ['<PAD>', '<START>', '<UNK>']:
                    continue
                
                generated_tokens.append(next_token)
                
                # Prepare next input
                current_token = torch.tensor([[next_idx]]).to(self.device)
            
            # Join tokens to form SMILES
            smiles = ''.join(generated_tokens)
            return smiles
    
    def generate_batch(self, num_samples=5000, temperature=0.8, 
                       validate=True, show_progress=True):
        """
        Generate a batch of molecules.
        
        Args:
            num_samples: Number of generation attempts
            temperature: Sampling temperature
            validate: Whether to validate with RDKit
            show_progress: Show progress bar
            
        Returns:
            Dictionary with results
        """
        generated = []
        valid_smiles = set()
        invalid_count = 0
        
        iterator = range(num_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating molecules", unit="mol")
        
        for _ in iterator:
            smiles = self.generate_smiles(temperature=temperature)
            
            if smiles:
                generated.append(smiles)
                
                if validate:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Canonicalize
                        canonical = Chem.MolToSmiles(mol, canonical=True)
                        valid_smiles.add(canonical)
                    else:
                        invalid_count += 1
                else:
                    valid_smiles.add(smiles)
        
        return {
            'generated': generated,
            'valid': list(valid_smiles),
            'num_attempts': num_samples,
            'num_valid': len(valid_smiles),
            'num_invalid': invalid_count,
            'validity_rate': len(valid_smiles) / len(generated) * 100 if generated else 0
        }


# ============================================================================
# MOLECULE FILTERS
# ============================================================================

def is_drug_like(smiles):
    """
    Check if molecule passes drug-likeness filters.
    Uses Lipinski's Rule of Five with some relaxations.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # Lipinski's Rule of Five
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        # Filters
        if not (200 <= mw <= 600):
            return False
        if not (-1 <= logp <= 5.5):
            return False
        if hbd > 5:
            return False
        if hba > 10:
            return False
        
        # Additional quality checks
        # Rotatable bonds
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        if rot_bonds > 10:
            return False
        
        # TPSA (Topological Polar Surface Area)
        tpsa = Descriptors.TPSA(mol)
        if tpsa > 140:
            return False
        
        return True
        
    except Exception:
        return False


def calculate_properties(smiles):
    """Calculate molecular properties for a valid SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            'smiles': smiles,
            'MW': round(Descriptors.MolWt(mol), 2),
            'LogP': round(Descriptors.MolLogP(mol), 2),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'TPSA': round(Descriptors.TPSA(mol), 2),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'Rings': Descriptors.RingCount(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol),
        }
    except Exception:
        return None


def check_novelty(generated_smiles, training_smiles):
    """
    Check how many generated molecules are novel (not in training set).
    
    Args:
        generated_smiles: List of generated SMILES
        training_smiles: Set of training SMILES
        
    Returns:
        List of novel SMILES, novelty rate
    """
    novel = [s for s in generated_smiles if s not in training_smiles]
    novelty_rate = len(novel) / len(generated_smiles) * 100 if generated_smiles else 0
    return novel, novelty_rate


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate molecules with trained LSTM')
    
    parser.add_argument('--model', type=str, default='finetune_best_model.pth',
                        help='Model file in models/ directory')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of generation attempts')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (0.5-1.2)')
    parser.add_argument('--output', type=str, default='generated_molecules.txt',
                        help='Output filename')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  PHASE 3: MOLECULE GENERATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.num_samples:,}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Output: {args.output}")
    
    # Paths
    model_path = MODELS_DIR / args.model
    vocab_path = MODELS_DIR / "pretrain_vocab.json"
    output_path = RESULTS_DIR / args.output
    
    # Check files exist
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        return
    if not vocab_path.exists():
        print(f"\n❌ Vocabulary not found: {vocab_path}")
        return
    
    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    generator = MoleculeGenerator(
        model_path=str(model_path),
        vocab_path=str(vocab_path)
    )
    
    # Generate molecules
    print(f"\n{'='*60}")
    print("GENERATING MOLECULES")
    print(f"{'='*60}")
    print(f"\n🧪 Generating {args.num_samples:,} molecule candidates...")
    print(f"   Temperature: {args.temperature} (creativity level)")
    
    results = generator.generate_batch(
        num_samples=args.num_samples,
        temperature=args.temperature,
        validate=True,
        show_progress=True
    )
    
    print(f"\n📊 Generation Results:")
    print(f"   Attempts:      {results['num_attempts']:,}")
    print(f"   Valid SMILES:  {results['num_valid']:,}")
    print(f"   Invalid:       {results['num_invalid']:,}")
    print(f"   Validity Rate: {results['validity_rate']:.1f}%")
    
    # Apply drug-likeness filters
    print(f"\n{'='*60}")
    print("FILTERING FOR DRUG-LIKENESS")
    print(f"{'='*60}")
    
    valid_smiles = results['valid']
    drug_like = [s for s in tqdm(valid_smiles, desc="Filtering") if is_drug_like(s)]
    
    print(f"\n💊 Drug-likeness Filter:")
    print(f"   Valid molecules: {len(valid_smiles):,}")
    print(f"   Drug-like:       {len(drug_like):,}")
    print(f"   Pass rate:       {len(drug_like)/len(valid_smiles)*100:.1f}%")
    
    # Check novelty against training data
    print(f"\n{'='*60}")
    print("CHECKING NOVELTY")
    print(f"{'='*60}")
    
    # Load training data
    training_file = DATA_DIR / "clean_smiles.txt"
    if training_file.exists():
        with open(training_file, 'r') as f:
            training_smiles = set(line.strip() for line in f)
        
        novel, novelty_rate = check_novelty(drug_like, training_smiles)
        
        print(f"\n🆕 Novelty Check:")
        print(f"   Training set size: {len(training_smiles):,}")
        print(f"   Novel molecules:   {len(novel):,}")
        print(f"   Novelty rate:      {novelty_rate:.1f}%")
    else:
        novel = drug_like
        print("   (Training file not found, skipping novelty check)")
    
    # Calculate properties for top molecules
    print(f"\n{'='*60}")
    print("ANALYZING TOP MOLECULES")
    print(f"{'='*60}")
    
    # Get properties for all novel drug-like molecules
    molecules_with_props = []
    for smiles in novel:
        props = calculate_properties(smiles)
        if props:
            molecules_with_props.append(props)
    
    # Sort by drug-likeness score (lower MW, moderate LogP)
    molecules_with_props.sort(key=lambda x: (abs(x['LogP'] - 2.5), x['MW']))
    
    print(f"\n🏆 Top 10 Generated Molecules:")
    print("-" * 80)
    print(f"{'SMILES':<50} {'MW':>8} {'LogP':>6} {'HBD':>4} {'HBA':>4}")
    print("-" * 80)
    
    for mol in molecules_with_props[:10]:
        smiles_short = mol['smiles'][:48] + '..' if len(mol['smiles']) > 50 else mol['smiles']
        print(f"{smiles_short:<50} {mol['MW']:>8.1f} {mol['LogP']:>6.2f} {mol['HBD']:>4} {mol['HBA']:>4}")
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save all valid novel drug-like molecules
    with open(output_path, 'w') as f:
        for smiles in novel:
            f.write(smiles + '\n')
    
    print(f"\n✓ Saved {len(novel):,} molecules to {output_path}")
    
    # Save detailed results with properties
    detailed_path = RESULTS_DIR / "generated_molecules_detailed.csv"
    with open(detailed_path, 'w') as f:
        f.write("SMILES,MW,LogP,HBD,HBA,TPSA,RotBonds,Rings,AromaticRings\n")
        for mol in molecules_with_props:
            f.write(f"{mol['smiles']},{mol['MW']},{mol['LogP']},{mol['HBD']},"
                    f"{mol['HBA']},{mol['TPSA']},{mol['RotBonds']},"
                    f"{mol['Rings']},{mol['AromaticRings']}\n")
    
    print(f"✓ Saved detailed properties to {detailed_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ PHASE 3 COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📈 Final Summary:")
    print(f"   Generation attempts: {args.num_samples:,}")
    print(f"   Valid SMILES:        {results['num_valid']:,} ({results['validity_rate']:.1f}%)")
    print(f"   Drug-like:           {len(drug_like):,}")
    print(f"   Novel & Drug-like:   {len(novel):,}")
    
    print(f"\n📁 Output files:")
    print(f"   - {output_path}")
    print(f"   - {detailed_path}")
    
    print(f"\n🎯 Next step: Phase 4 (Evaluation)")
    print(f"   Dock these molecules against EGFR to find the best candidates!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
