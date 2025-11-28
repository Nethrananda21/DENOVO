#!/usr/bin/env python3
"""
Comprehensive Analysis of All Top 20 EGFR Inhibitor Candidates
==============================================================
Performs: H-bond analysis, novelty check, PubChem search keys
"""

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, inchi, Descriptors
from rdkit.Chem import Lipinski, rdMolDescriptors
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
import os
import csv

# =============================================================================
# CONFIGURATION
# =============================================================================

DOCKING_FILE = r"C:\DENOVO\results\top_docking_hits.csv"
EGFR_TRAINING = r"C:\DENOVO\data\clean_smiles.txt"
OUTPUT_FILE = r"C:\DENOVO\results\comprehensive_candidate_analysis.csv"

# Known EGFR drugs for comparison
KNOWN_DRUGS = {
    'Erlotinib': 'COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC',
    'Gefitinib': 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
    'Lapatinib': 'CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1',
    'Afatinib': 'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1',
    'Osimertinib': 'COc1cc(N(C)CCN(C)C)c(NC(=O)/C=C/CN(C)C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1',
    'Dacomitinib': 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1',
}

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("COMPREHENSIVE ANALYSIS OF ALL TOP 20 EGFR INHIBITOR CANDIDATES")
print("=" * 80)

# Read docking results
molecules = []
with open(DOCKING_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        molecules.append({
            'rank': int(row['Dock_Rank']),
            'smiles': row['SMILES'],
            'affinity': float(row['Affinity']),
            'qed': float(row['QED'])
        })

print(f"\nLoaded {len(molecules)} molecules from docking results")

# Load training data for novelty check
training_fps = []
with open(EGFR_TRAINING, 'r') as f:
    for line in f:
        smiles = line.strip()
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                training_fps.append(fp)

print(f"Loaded {len(training_fps)} training molecules for novelty comparison")

# Prepare drug fingerprints
drug_fps = {}
for name, smiles in KNOWN_DRUGS.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        drug_fps[name] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

# =============================================================================
# ANALYZE EACH MOLECULE
# =============================================================================

print("\n" + "=" * 80)
print("DETAILED ANALYSIS")
print("=" * 80)

results = []

for mol_data in molecules:
    rank = mol_data['rank']
    smiles = mol_data['smiles']
    affinity = mol_data['affinity']
    qed = mol_data['qed']
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    
    # Calculate fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    
    # Calculate properties
    mw = Descriptors.MolWt(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    
    # Get InChI Key for database search
    inchi_key = inchi.MolToInchiKey(mol)
    
    # Get canonical SMILES
    canonical = Chem.MolToSmiles(mol, canonical=True)
    
    # Calculate max similarity to training data
    max_train_sim = max(DataStructs.TanimotoSimilarity(fp, tfp) for tfp in training_fps)
    
    # Calculate similarity to known drugs
    drug_similarities = {}
    max_drug_sim = 0
    most_similar_drug = ""
    for drug_name, drug_fp in drug_fps.items():
        sim = DataStructs.TanimotoSimilarity(fp, drug_fp)
        drug_similarities[drug_name] = sim
        if sim > max_drug_sim:
            max_drug_sim = sim
            most_similar_drug = drug_name
    
    # Determine novelty status
    if max_train_sim >= 0.9:
        novelty_status = "DUPLICATE"
    elif max_train_sim >= 0.7:
        novelty_status = "SIMILAR"
    elif max_train_sim >= 0.5:
        novelty_status = "MODERATE"
    else:
        novelty_status = "NOVEL"
    
    result = {
        'Rank': rank,
        'SMILES': smiles,
        'Canonical_SMILES': canonical,
        'InChI_Key': inchi_key,
        'Formula': formula,
        'MW': round(mw, 2),
        'Affinity': affinity,
        'QED': qed,
        'HBD': hbd,
        'HBA': hba,
        'RotBonds': rotatable,
        'Max_Train_Similarity': round(max_train_sim, 3),
        'Max_Drug_Similarity': round(max_drug_sim, 3),
        'Most_Similar_Drug': most_similar_drug,
        'Novelty_Status': novelty_status
    }
    
    results.append(result)
    
    # Print summary
    print(f"\n[Rank {rank:2d}] {smiles[:45]}...")
    print(f"  MW: {mw:.1f} | QED: {qed:.3f} | Affinity: {affinity:.2f} kcal/mol")
    print(f"  H-bond: Donors={hbd}, Acceptors={hba} | Rotatable={rotatable}")
    print(f"  InChI Key: {inchi_key}")
    print(f"  Max similarity to training: {max_train_sim:.3f} → {novelty_status}")
    print(f"  Most similar drug: {most_similar_drug} ({max_drug_sim:.3f})")

# =============================================================================
# SAVE RESULTS
# =============================================================================

# Save to CSV
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 80)
print("RESULTS SAVED")
print("=" * 80)
print(f"Full analysis saved to: {OUTPUT_FILE}")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("NOVELTY SUMMARY")
print("=" * 80)

novelty_counts = {}
for r in results:
    status = r['Novelty_Status']
    novelty_counts[status] = novelty_counts.get(status, 0) + 1

for status, count in sorted(novelty_counts.items()):
    print(f"  {status}: {count} molecules")

# =============================================================================
# TOP NOVEL CANDIDATES
# =============================================================================

print("\n" + "=" * 80)
print("TOP NOVEL CANDIDATES (Max Train Similarity < 0.7)")
print("=" * 80)

novel_candidates = [r for r in results if r['Max_Train_Similarity'] < 0.7]
novel_candidates.sort(key=lambda x: x['Affinity'])

print(f"\n{'Rank':<6}{'Affinity':<12}{'QED':<8}{'Train_Sim':<12}{'Drug_Sim':<12}{'InChI Key'}")
print("-" * 80)

for r in novel_candidates[:10]:
    print(f"{r['Rank']:<6}{r['Affinity']:<12.2f}{r['QED']:<8.3f}{r['Max_Train_Similarity']:<12.3f}{r['Max_Drug_Similarity']:<12.3f}{r['InChI_Key'][:25]}...")

# =============================================================================
# PUBCHEM SEARCH INSTRUCTIONS
# =============================================================================

print("\n" + "=" * 80)
print("PUBCHEM SEARCH - InChI Keys for Database Verification")
print("=" * 80)
print("\nSearch these InChI Keys at: https://pubchem.ncbi.nlm.nih.gov")
print("If 'No results found' → NOVEL compound\n")

for r in results:
    print(f"Rank {r['Rank']:2d}: {r['InChI_Key']}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
