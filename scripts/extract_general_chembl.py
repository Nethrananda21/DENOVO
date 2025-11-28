"""
Extract 50,000 General Valid SMILES from ChEMBL
===============================================
Alternative approach: Use direct REST API calls instead of the Python client.
This avoids the filter bug in the ChEMBL web resource client.

Purpose: Build a "general chemistry brain" before specializing on EGFR.
"""

import os
import re
import sys
import json
import random
import requests
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from tqdm import tqdm

# Configuration
TARGET_COUNT = 50000  # Target number of valid SMILES
BATCH_SIZE = 1000     # Fetch in batches (API limit)
MAX_SMILES_LENGTH = 100  # Keep reasonably sized molecules

# Output paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "general_chembl.txt"

# ChEMBL REST API base URL
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def is_valid_druglike(smiles: str) -> bool:
    """
    Check if SMILES represents a valid, drug-like molecule.
    Uses Lipinski's Rule of Five with relaxed constraints.
    """
    if not smiles or len(smiles) > MAX_SMILES_LENGTH:
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # Basic Lipinski-like filters (relaxed for diversity)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        # Relaxed drug-likeness criteria
        if not (150 <= mw <= 600):
            return False
        if not (-2 <= logp <= 6):
            return False
        if hbd > 6:
            return False
        if hba > 12:
            return False
        
        # No metals or unusual elements
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        allowed_atoms = {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I', 'H'}
        if not all(a in allowed_atoms for a in atoms):
            return False
        
        # At least some carbon atoms (organic)
        if atoms.count('C') < 3:
            return False
        
        return True
        
    except Exception:
        return False


def standardize_smiles(smiles: str) -> str:
    """Standardize SMILES to canonical form."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def fetch_molecules_via_rest_api(offset=0, limit=1000):
    """
    Fetch molecules from ChEMBL using direct REST API.
    This bypasses the buggy Python client filter method.
    """
    url = f"{CHEMBL_API_BASE}/molecule.json"
    params = {
        'limit': limit,
        'offset': offset,
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get('molecules', [])
    except Exception as e:
        print(f"   API request failed: {e}")
        return []


def fetch_general_molecules():
    """
    Fetch diverse drug-like molecules from ChEMBL using REST API.
    """
    print("=" * 60)
    print("GENERAL ChEMBL SMILES EXTRACTION")
    print("=" * 60)
    print(f"Target: {TARGET_COUNT:,} valid drug-like molecules")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)
    
    collected_smiles = set()
    offset = 0
    attempts = 0
    max_attempts = 200  # Safety limit (200 * 1000 = 200k molecules scanned)
    
    print("\n📡 Fetching molecules from ChEMBL REST API...")
    print("   (This may take several minutes)\n")
    
    with tqdm(total=TARGET_COUNT, desc="Collecting valid SMILES", unit="mol") as pbar:
        while len(collected_smiles) < TARGET_COUNT and attempts < max_attempts:
            attempts += 1
            
            # Fetch batch via REST API
            molecules = fetch_molecules_via_rest_api(offset=offset, limit=BATCH_SIZE)
            
            if not molecules:
                print(f"\n⚠️  No more results at offset {offset}")
                break
            
            # Process each molecule
            valid_count = 0
            for mol_data in molecules:
                if len(collected_smiles) >= TARGET_COUNT:
                    break
                
                try:
                    # Extract SMILES from molecule_structures
                    structures = mol_data.get('molecule_structures')
                    if structures:
                        smiles = structures.get('canonical_smiles')
                        if smiles and is_valid_druglike(smiles):
                            std_smiles = standardize_smiles(smiles)
                            if std_smiles and std_smiles not in collected_smiles:
                                collected_smiles.add(std_smiles)
                                valid_count += 1
                                pbar.update(1)
                except Exception:
                    continue
            
            offset += BATCH_SIZE
            
            # Status update
            if attempts % 10 == 0:
                tqdm.write(f"   Scanned {offset:,} molecules, collected {len(collected_smiles):,} valid")
    
    print(f"\n✅ Collection complete: {len(collected_smiles):,} unique valid SMILES")
    
    return list(collected_smiles)


def save_smiles(smiles_list: list):
    """Save collected SMILES to file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Shuffle for training randomness
    random.shuffle(smiles_list)
    
    with open(OUTPUT_FILE, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + '\n')
    
    print(f"\n💾 Saved {len(smiles_list):,} SMILES to {OUTPUT_FILE}")
    
    # Statistics
    lengths = [len(s) for s in smiles_list]
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total molecules:   {len(smiles_list):,}")
    print(f"   Avg SMILES length: {sum(lengths)/len(lengths):.1f}")
    print(f"   Min SMILES length: {min(lengths)}")
    print(f"   Max SMILES length: {max(lengths)}")
    
    # Examples
    print(f"\n📝 Sample SMILES:")
    for i, smiles in enumerate(smiles_list[:5]):
        print(f"   {i+1}. {smiles}")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("  GENERAL ChEMBL EXTRACTION FOR PRE-TRAINING")
    print("="*60)
    print("\nThis script will fetch ~50,000 diverse drug-like molecules")
    print("from ChEMBL to train a general 'chemistry language' model.\n")
    
    # Check if file already exists
    if OUTPUT_FILE.exists():
        existing_count = sum(1 for _ in open(OUTPUT_FILE))
        print(f"⚠️  File already exists with {existing_count:,} molecules")
        response = input("   Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("   Aborted.")
            return
    
    # Fetch molecules
    smiles_list = fetch_general_molecules()
    
    if len(smiles_list) < 1000:
        print(f"\n❌ Only collected {len(smiles_list)} molecules. Something went wrong.")
        return
    
    # Save to file
    save_smiles(smiles_list)
    
    print("\n" + "="*60)
    print("✅ EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\nNext step: Run pre-training with:")
    print(f"   python scripts/phase2_train_transfer.py --mode pretrain")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
