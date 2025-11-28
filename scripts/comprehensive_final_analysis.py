#!/usr/bin/env python3
"""
Comprehensive Analysis - All 20 EGFR Inhibitor Candidates
==========================================================
1. Extract actual Vina affinities from all logs
2. Calculate InChI Keys for PubChem search
3. Generate comprehensive summary
"""

import os
import re
import csv
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import inchi, Descriptors, AllChem, DataStructs
from rdkit.Chem import Lipinski, rdMolDescriptors
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

# =============================================================================
# CONFIGURATION
# =============================================================================

DOCKING_FILE = Path(r"C:\DENOVO\results\top_docking_hits.csv")
ORIGINAL_LOGS = Path(r"C:\DENOVO\docking\vina_results\logs")
EXTENDED_LOGS = Path(r"C:\DENOVO\docking\vina_results_extended\logs")
OUTPUT_DIR = Path(r"C:\DENOVO\results")

print("=" * 80)
print("COMPREHENSIVE ANALYSIS - ALL 20 EGFR INHIBITOR CANDIDATES")
print("=" * 80)

# =============================================================================
# PARSE VINA LOGS
# =============================================================================

def parse_vina_log(log_path):
    """Extract best affinity from Vina log file."""
    if not log_path.exists():
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Look for affinity in results table
    pattern = r'^\s*1\s+(-?\d+\.?\d*)'
    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            return float(match.group(1))
    
    return None

# Collect all results
all_results = []

# Load molecule info from docking file
molecules = {}
with open(DOCKING_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rank = int(row['Dock_Rank'])
        molecules[rank] = {
            'smiles': row['SMILES'],
            'estimated': float(row['Affinity']),
            'qed': float(row['QED'])
        }

print(f"\nLoaded {len(molecules)} molecules from docking file")

# Get Vina results from logs
print("\nExtracting Vina affinities from logs...")

for rank in range(1, 21):
    mol_data = molecules.get(rank)
    if not mol_data:
        continue
    
    mol_id = f"mol_{rank:03d}"
    smiles = mol_data['smiles']
    
    # Try original logs first, then extended
    log_path = ORIGINAL_LOGS / f"{mol_id}_vina.log"
    if not log_path.exists():
        log_path = EXTENDED_LOGS / f"{mol_id}_vina.log"
    
    vina_affinity = parse_vina_log(log_path)
    
    # Calculate molecular properties
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    
    # InChI Key
    inchi_key = inchi.MolToInchiKey(mol)
    
    # Properties
    mw = Descriptors.MolWt(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)
    
    result = {
        'Rank': rank,
        'Mol_ID': mol_id,
        'SMILES': smiles,
        'InChI_Key': inchi_key,
        'Estimated_Affinity': mol_data['estimated'],
        'Vina_Affinity': vina_affinity,
        'QED': mol_data['qed'],
        'MW': round(mw, 1),
        'HBD': hbd,
        'HBA': hba,
        'RotBonds': rotatable,
        'TPSA': round(tpsa, 1),
        'LogP': round(logp, 2)
    }
    
    all_results.append(result)
    
    status = f"{vina_affinity:.2f} kcal/mol" if vina_affinity else "NO DATA"
    print(f"  {mol_id}: {status}")

# =============================================================================
# SORT BY VINA AFFINITY
# =============================================================================

# Filter out molecules without Vina data and sort
valid_results = [r for r in all_results if r['Vina_Affinity'] is not None]
valid_results.sort(key=lambda x: x['Vina_Affinity'])

# =============================================================================
# SAVE COMPREHENSIVE RESULTS
# =============================================================================

output_file = OUTPUT_DIR / 'all_20_comprehensive_analysis.csv'
with open(output_file, 'w', newline='') as f:
    fieldnames = ['Vina_Rank', 'Original_Rank', 'Mol_ID', 'SMILES', 'InChI_Key', 
                  'Vina_Affinity', 'Estimated_Affinity', 'QED', 'MW', 'HBD', 'HBA', 
                  'RotBonds', 'TPSA', 'LogP', 'Classification']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    
    for i, r in enumerate(valid_results, 1):
        aff = r['Vina_Affinity']
        if aff < -9.5:
            cls = "EXCELLENT"
        elif aff < -8.5:
            cls = "Good"
        elif aff < -7.0:
            cls = "Moderate"
        else:
            cls = "Weak"
        
        writer.writerow({
            'Vina_Rank': i,
            'Original_Rank': r['Rank'],
            'Mol_ID': r['Mol_ID'],
            'SMILES': r['SMILES'],
            'InChI_Key': r['InChI_Key'],
            'Vina_Affinity': r['Vina_Affinity'],
            'Estimated_Affinity': r['Estimated_Affinity'],
            'QED': r['QED'],
            'MW': r['MW'],
            'HBD': r['HBD'],
            'HBA': r['HBA'],
            'RotBonds': r['RotBonds'],
            'TPSA': r['TPSA'],
            'LogP': r['LogP'],
            'Classification': cls
        })

print(f"\n✓ Saved comprehensive results to: {output_file}")

# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 100)
print("ALL 20 MOLECULES - RANKED BY ACTUAL VINA DOCKING AFFINITY")
print("=" * 100)

print(f"\n{'Vina':>5} {'Orig':>5} {'Mol ID':<10} {'Vina Aff':>10} {'Est Aff':>10} {'QED':>6} {'MW':>7} {'Classification':<12} {'InChI Key':<28}")
print("-" * 100)

for i, r in enumerate(valid_results, 1):
    aff = r['Vina_Affinity']
    if aff < -9.5:
        cls = "EXCELLENT"
    elif aff < -8.5:
        cls = "Good"
    elif aff < -7.0:
        cls = "Moderate"
    else:
        cls = "Weak"
    
    print(f"{i:>5} {r['Rank']:>5} {r['Mol_ID']:<10} {aff:>10.2f} {r['Estimated_Affinity']:>10.2f} {r['QED']:>6.3f} {r['MW']:>7.1f} {cls:<12} {r['InChI_Key'][:28]}")

# =============================================================================
# TOP 10 SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("TOP 10 CANDIDATES - PUBCHEM SEARCH KEYS")
print("=" * 80)
print("\nSearch these InChI Keys at: https://pubchem.ncbi.nlm.nih.gov")
print("If 'No results found' → NOVEL compound\n")

for i, r in enumerate(valid_results[:10], 1):
    print(f"{i:2}. {r['Mol_ID']} (Vina: {r['Vina_Affinity']:.2f} kcal/mol)")
    print(f"    InChI Key: {r['InChI_Key']}")
    print(f"    SMILES: {r['SMILES'][:60]}...")
    print()

# =============================================================================
# STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("CLASSIFICATION SUMMARY")
print("=" * 80)

classifications = {}
for r in valid_results:
    aff = r['Vina_Affinity']
    if aff < -9.5:
        cls = "EXCELLENT (< -9.5)"
    elif aff < -8.5:
        cls = "Good (-8.5 to -9.5)"
    elif aff < -7.0:
        cls = "Moderate (-7.0 to -8.5)"
    else:
        cls = "Weak (> -7.0)"
    
    classifications[cls] = classifications.get(cls, 0) + 1

for cls, count in sorted(classifications.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {count} molecules")

# Average affinity
avg_aff = sum(r['Vina_Affinity'] for r in valid_results) / len(valid_results)
print(f"\n  Average Vina Affinity: {avg_aff:.2f} kcal/mol")

# Best molecule
best = valid_results[0]
print(f"  Best Molecule: {best['Mol_ID']} at {best['Vina_Affinity']:.2f} kcal/mol")

# =============================================================================
# DRUG-LIKENESS FILTER
# =============================================================================

print("\n" + "=" * 80)
print("DRUG-LIKENESS FILTER (Lipinski's Rule of 5)")
print("=" * 80)

lipinski_pass = []
for r in valid_results:
    violations = 0
    if r['MW'] > 500: violations += 1
    if r['HBD'] > 5: violations += 1
    if r['HBA'] > 10: violations += 1
    if r['LogP'] > 5: violations += 1
    
    if violations <= 1:
        lipinski_pass.append(r)
        status = "PASS" if violations == 0 else f"PASS ({violations} violation)"
    else:
        status = f"FAIL ({violations} violations)"

print(f"\nPassing Lipinski's Rule: {len(lipinski_pass)}/{len(valid_results)} molecules")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
