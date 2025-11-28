"""
Phase 4: Evaluation Funnel
==========================
Multi-stage filtering to find the best drug candidates.

Step 1: Novelty & Scaffold Filter (Tanimoto < 0.6)
Step 2: ADMET Filter (QED > 0.6)

Usage:
    python scripts/phase4_evaluate.py
"""

import os
import sys
from pathlib import Path
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from tqdm import tqdm

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================================
# STEP 1: NOVELTY & SCAFFOLD FILTER
# ============================================================================

def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Get Morgan fingerprint for a molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except:
        return None


def get_scaffold(smiles):
    """Extract Murcko scaffold from molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except:
        return None


def calculate_max_tanimoto(fp, reference_fps):
    """Calculate maximum Tanimoto similarity to any reference molecule."""
    if fp is None:
        return 1.0  # Treat as too similar (will be filtered out)
    
    max_sim = 0.0
    for ref_fp in reference_fps:
        sim = DataStructs.TanimotoSimilarity(fp, ref_fp)
        if sim > max_sim:
            max_sim = sim
    return max_sim


def novelty_scaffold_filter(generated_smiles, training_smiles, tanimoto_threshold=0.6):
    """
    Filter molecules based on novelty (Tanimoto similarity) and scaffold diversity.
    
    Args:
        generated_smiles: List of generated SMILES
        training_smiles: List of training SMILES
        tanimoto_threshold: Maximum allowed similarity (default 0.6)
        
    Returns:
        List of novel molecules, statistics dict
    """
    print("\n" + "="*60)
    print("STEP 1: NOVELTY & SCAFFOLD FILTER")
    print("="*60)
    print(f"\nInput: {len(generated_smiles):,} molecules")
    print(f"Tanimoto threshold: < {tanimoto_threshold}")
    
    # Calculate fingerprints for training set
    print("\n📊 Computing training set fingerprints...")
    training_fps = []
    for smiles in tqdm(training_smiles, desc="Training FPs"):
        fp = get_morgan_fingerprint(smiles)
        if fp is not None:
            training_fps.append(fp)
    print(f"   Training fingerprints: {len(training_fps):,}")
    
    # Calculate fingerprints and similarity for generated molecules
    print("\n🔍 Analyzing generated molecules...")
    novel_molecules = []
    similarities = []
    scaffolds_seen = set()
    scaffold_counts = Counter()
    
    for smiles in tqdm(generated_smiles, desc="Filtering"):
        fp = get_morgan_fingerprint(smiles)
        if fp is None:
            continue
        
        # Calculate max Tanimoto similarity
        max_sim = calculate_max_tanimoto(fp, training_fps)
        similarities.append(max_sim)
        
        # Filter by Tanimoto threshold
        if max_sim < tanimoto_threshold:
            # Also track scaffold diversity
            scaffold = get_scaffold(smiles)
            if scaffold:
                scaffold_counts[scaffold] += 1
            
            novel_molecules.append({
                'smiles': smiles,
                'tanimoto': max_sim,
                'scaffold': scaffold
            })
    
    # Statistics
    stats = {
        'input_count': len(generated_smiles),
        'output_count': len(novel_molecules),
        'pass_rate': len(novel_molecules) / len(generated_smiles) * 100,
        'avg_tanimoto': sum(similarities) / len(similarities) if similarities else 0,
        'unique_scaffolds': len(scaffold_counts),
    }
    
    print(f"\n📈 Novelty Filter Results:")
    print(f"   Input molecules:    {stats['input_count']:,}")
    print(f"   Novel (Tan < {tanimoto_threshold}): {stats['output_count']:,}")
    print(f"   Pass rate:          {stats['pass_rate']:.1f}%")
    print(f"   Avg Tanimoto:       {stats['avg_tanimoto']:.3f}")
    print(f"   Unique scaffolds:   {stats['unique_scaffolds']:,}")
    
    # Show Tanimoto distribution
    if similarities:
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        print(f"\n   Tanimoto Distribution:")
        for i in range(len(bins)-1):
            count = sum(1 for s in similarities if bins[i] <= s < bins[i+1])
            pct = count / len(similarities) * 100
            bar = "█" * int(pct / 2)
            print(f"   {bins[i]:.1f}-{bins[i+1]:.1f}: {bar} {count:,} ({pct:.1f}%)")
    
    return novel_molecules, stats


# ============================================================================
# STEP 2: ADMET / QED FILTER
# ============================================================================

def calculate_qed(smiles):
    """Calculate QED (Quantitative Estimate of Drug-likeness)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        return QED.qed(mol)
    except:
        return 0.0


def calculate_admet_properties(smiles):
    """Calculate ADMET-related properties."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'Rings': Descriptors.RingCount(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol),
            'QED': QED.qed(mol),
        }
    except:
        return None


def admet_qed_filter(molecules, qed_threshold=0.6):
    """
    Filter molecules based on QED score.
    
    Args:
        molecules: List of molecule dicts with 'smiles' key
        qed_threshold: Minimum QED score (default 0.6)
        
    Returns:
        List of high-quality molecules, statistics dict
    """
    print("\n" + "="*60)
    print("STEP 2: ADMET / QED FILTER")
    print("="*60)
    print(f"\nInput: {len(molecules):,} molecules")
    print(f"QED threshold: > {qed_threshold}")
    
    high_quality = []
    qed_scores = []
    
    for mol_dict in tqdm(molecules, desc="Calculating QED"):
        smiles = mol_dict['smiles']
        qed_score = calculate_qed(smiles)
        qed_scores.append(qed_score)
        
        if qed_score > qed_threshold:
            props = calculate_admet_properties(smiles)
            if props:
                high_quality.append({
                    'smiles': smiles,
                    'tanimoto': mol_dict['tanimoto'],
                    'scaffold': mol_dict['scaffold'],
                    **props
                })
    
    # Sort by QED score (best first)
    high_quality.sort(key=lambda x: x['QED'], reverse=True)
    
    # Statistics
    stats = {
        'input_count': len(molecules),
        'output_count': len(high_quality),
        'pass_rate': len(high_quality) / len(molecules) * 100 if molecules else 0,
        'avg_qed': sum(qed_scores) / len(qed_scores) if qed_scores else 0,
        'max_qed': max(qed_scores) if qed_scores else 0,
    }
    
    print(f"\n📈 QED Filter Results:")
    print(f"   Input molecules:     {stats['input_count']:,}")
    print(f"   High QED (> {qed_threshold}):   {stats['output_count']:,}")
    print(f"   Pass rate:           {stats['pass_rate']:.1f}%")
    print(f"   Avg QED:             {stats['avg_qed']:.3f}")
    print(f"   Max QED:             {stats['max_qed']:.3f}")
    
    # Show QED distribution
    if qed_scores:
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        print(f"\n   QED Distribution:")
        for i in range(len(bins)-1):
            count = sum(1 for s in qed_scores if bins[i] <= s < bins[i+1])
            pct = count / len(qed_scores) * 100
            bar = "█" * int(pct / 2)
            print(f"   {bins[i]:.1f}-{bins[i+1]:.1f}: {bar} {count:,} ({pct:.1f}%)")
    
    return high_quality, stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("  PHASE 4: EVALUATION FUNNEL")
    print("="*60)
    
    # Load generated molecules
    generated_file = RESULTS_DIR / "generated_molecules.txt"
    if not generated_file.exists():
        print(f"\n❌ Generated molecules not found: {generated_file}")
        print("   Run phase3_generate.py first!")
        return
    
    with open(generated_file, 'r') as f:
        generated_smiles = [line.strip() for line in f if line.strip()]
    
    print(f"\n✓ Loaded {len(generated_smiles):,} generated molecules")
    
    # Load training data for novelty check
    training_file = DATA_DIR / "clean_smiles.txt"
    if not training_file.exists():
        print(f"\n❌ Training data not found: {training_file}")
        return
    
    with open(training_file, 'r') as f:
        training_smiles = [line.strip() for line in f if line.strip()]
    
    print(f"✓ Loaded {len(training_smiles):,} training molecules")
    
    # =========================================================================
    # STEP 1: Novelty & Scaffold Filter
    # =========================================================================
    novel_molecules, novelty_stats = novelty_scaffold_filter(
        generated_smiles, 
        training_smiles, 
        tanimoto_threshold=0.6
    )
    
    # =========================================================================
    # STEP 2: ADMET / QED Filter
    # =========================================================================
    final_candidates, qed_stats = admet_qed_filter(
        novel_molecules, 
        qed_threshold=0.6
    )
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "="*60)
    print("SAVING FINAL CANDIDATES")
    print("="*60)
    
    # Save final candidates
    output_file = RESULTS_DIR / "final_candidates.csv"
    with open(output_file, 'w') as f:
        f.write("Rank,SMILES,QED,Tanimoto,MW,LogP,TPSA,HBD,HBA,RotBonds,Scaffold\n")
        for i, mol in enumerate(final_candidates, 1):
            scaffold = mol['scaffold'] if mol['scaffold'] else ''
            f.write(f"{i},{mol['smiles']},{mol['QED']:.3f},{mol['tanimoto']:.3f},"
                    f"{mol['MW']:.1f},{mol['LogP']:.2f},{mol['TPSA']:.1f},"
                    f"{mol['HBD']},{mol['HBA']},{mol['RotBonds']},{scaffold}\n")
    
    print(f"\n✓ Saved {len(final_candidates)} candidates to {output_file}")
    
    # Also save just SMILES for easy use
    smiles_file = RESULTS_DIR / "final_candidates_smiles.txt"
    with open(smiles_file, 'w') as f:
        for mol in final_candidates:
            f.write(mol['smiles'] + '\n')
    
    print(f"✓ Saved SMILES to {smiles_file}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("🏆 EVALUATION FUNNEL SUMMARY")
    print("="*60)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │  STAGE                          INPUT  →  OUTPUT       │
    ├─────────────────────────────────────────────────────────┤
    │  Generated Molecules            -      →  {len(generated_smiles):,}         │
    │  Step 1: Novelty (Tan < 0.6)    {len(generated_smiles):,}  →  {len(novel_molecules):,}           │
    │  Step 2: QED > 0.6              {len(novel_molecules):,}    →  {len(final_candidates):,}             │
    └─────────────────────────────────────────────────────────┘
    """)
    
    if final_candidates:
        print("\n🏆 TOP 10 DRUG CANDIDATES:")
        print("-" * 90)
        print(f"{'Rank':<5} {'QED':<6} {'Tan':<6} {'MW':<8} {'LogP':<7} {'SMILES':<50}")
        print("-" * 90)
        
        for i, mol in enumerate(final_candidates[:10], 1):
            smiles_short = mol['smiles'][:48] + '..' if len(mol['smiles']) > 50 else mol['smiles']
            print(f"{i:<5} {mol['QED']:<6.3f} {mol['tanimoto']:<6.3f} "
                  f"{mol['MW']:<8.1f} {mol['LogP']:<7.2f} {smiles_short}")
        
        print("-" * 90)
    
    print(f"\n✅ PHASE 4 COMPLETE!")
    print(f"\n📁 Output files:")
    print(f"   - {output_file}")
    print(f"   - {smiles_file}")
    print(f"\n🎯 These {len(final_candidates)} molecules are your top EGFR inhibitor candidates!")
    print("   Ready for molecular docking or further computational analysis.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
