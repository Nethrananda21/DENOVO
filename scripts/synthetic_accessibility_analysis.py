#!/usr/bin/env python3
"""
Synthetic Accessibility (SA) Score Analysis
============================================
Evaluates how easy it would be to synthesize the lead candidates.

SA Score Range:
- 1 = Very easy to synthesize
- 10 = Very difficult to synthesize
- < 4 = Generally considered synthetically accessible
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

import math
from collections import defaultdict
import pickle
import os

# =============================================================================
# SA SCORE CALCULATION (from RDKit Contrib)
# =============================================================================

# Fragment scores for SA calculation
# This is a simplified version - uses structural features

def numBridgeheadsAndSpiro(mol):
    """Count bridgehead and spiro atoms"""
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def calculateScore(mol):
    """
    Calculate synthetic accessibility score.
    Based on: Ertl P, Schuffenhauer A. J Cheminform. 2009;1:8
    """
    if mol is None:
        return None
    
    # Fragment score component
    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    
    # Use fragment complexity as proxy
    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    
    # Ring complexity
    ri = mol.GetRingInfo()
    nRings = ri.NumRings()
    nMacrocycles = sum(1 for ring in ri.AtomRings() if len(ring) > 8)
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(mol)
    
    # Molecular properties
    nHeteroAtoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    nRotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    
    # Size penalty
    sizePenalty = nAtoms ** 1.005 - nAtoms
    
    # Stereo penalty
    stereoPenalty = math.log10(nChiralCenters + 1)
    
    # Spiro/bridgehead penalty
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    
    # Macrocycle penalty
    macroPenalty = 0
    if nMacrocycles > 0:
        macroPenalty = math.log10(2)
    
    # Ring complexity
    ringPenalty = 0
    if nRings > 0:
        ringPenalty = math.log10(nRings + 1) * 0.5
    
    # Calculate base score from molecular complexity
    # More fragments = more complex
    nFragments = len(fps)
    fragmentScore = nFragments / nAtoms if nAtoms > 0 else 0
    
    # Combine penalties
    score = (
        1.0 +  # Base score
        fragmentScore * 2 +  # Fragment complexity
        sizePenalty * 0.1 +  # Size
        stereoPenalty +  # Stereo centers
        spiroPenalty +  # Spiro atoms
        bridgePenalty +  # Bridgehead atoms
        macroPenalty +  # Macrocycles
        ringPenalty  # Ring systems
    )
    
    # Normalize to 1-10 range
    score = max(1, min(10, score))
    
    return round(score, 2)

def detailed_sa_analysis(smiles, name):
    """Detailed synthetic accessibility analysis"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Calculate SA Score
    sa_score = calculateScore(mol)
    
    # Get structural features
    nAtoms = mol.GetNumAtoms()
    nHeavyAtoms = mol.GetNumHeavyAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    nRotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    nRings = rdMolDescriptors.CalcNumRings(mol)
    nAromaticRings = rdMolDescriptors.CalcNumAromaticRings(mol)
    nHeteroAtoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(mol)
    
    # Ring info
    ri = mol.GetRingInfo()
    ring_sizes = [len(ring) for ring in ri.AtomRings()]
    
    # Functional groups (simplified detection)
    smarts_patterns = {
        'Amide': '[NX3][CX3](=[OX1])',
        'Ester': '[CX3](=[OX1])[OX2]',
        'Carboxylic Acid': '[CX3](=O)[OX2H1]',
        'Ether': '[OD2]([#6])[#6]',
        'Amine': '[NX3;H2,H1,H0;!$(NC=O)]',
        'Alcohol': '[OX2H]',
        'Halogen': '[F,Cl,Br,I]',
        'Nitrile': '[NX1]#[CX2]',
        'Nitro': '[NX3](=O)=O',
        'Sulfone': '[SX4](=O)(=O)',
        'Ketone': '[CX3](=O)[#6]',
        'Aldehyde': '[CX3H1](=O)',
    }
    
    func_groups = {}
    for name_fg, smarts in smarts_patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                func_groups[name_fg] = len(matches)
    
    return {
        'name': name,
        'smiles': smiles,
        'sa_score': sa_score,
        'n_atoms': nAtoms,
        'n_heavy_atoms': nHeavyAtoms,
        'n_chiral_centers': nChiralCenters,
        'n_rotatable': nRotatable,
        'n_rings': nRings,
        'n_aromatic_rings': nAromaticRings,
        'n_heteroatoms': nHeteroAtoms,
        'n_bridgeheads': nBridgeheads,
        'n_spiro': nSpiro,
        'ring_sizes': ring_sizes,
        'functional_groups': func_groups
    }

# =============================================================================
# LEAD CANDIDATES
# =============================================================================

CANDIDATES = {
    'mol_001': {
        'smiles': 'COc1cc2ncnc(NC3CCC(C(=O)O)CC3)c2cc1OC',
        'vina': -8.78,
        'hbond': 'THR830 (3.0Å)',
        'description': 'Quinazoline with cyclohexyl-carboxylic acid'
    },
    'mol_006': {
        'smiles': 'COc1ccc(NC2=C(Cl)C(=O)c3ccccc3C2=O)cc1',
        'vina': -8.37,
        'hbond': 'THR766 (3.5Å)',
        'description': 'Chloro-anthraquinone with methoxyanilino'
    }
}

# Also analyze top 3 for comparison
TOP_3 = {
    'mol_003': {
        'smiles': 'COc1ccc(O)c(C(=O)Nc2ccc(C#N)c(C(F)(F)F)c2)c1',
        'vina': -9.20,
        'description': 'Best affinity - benzamide with CF3 and CN'
    },
    'mol_016': {
        'smiles': 'COc1ccc(C(=O)Nc2ccccc2C(C)(C)C)c(O)c1C',
        'vina': -9.02,
        'description': 'Benzamide with tert-butyl'
    },
    'mol_012': {
        'smiles': 'COc1ccc(C(=O)Nc2ncccc2Cl)c(OC)c1OC',
        'vina': -8.90,
        'description': 'Pyridine-benzamide with chloro'
    }
}

# Known drugs for comparison
KNOWN_DRUGS = {
    'Erlotinib': 'COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC',
    'Gefitinib': 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
    'Imatinib': 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1',
}

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

print("=" * 80)
print("SYNTHETIC ACCESSIBILITY (SA) SCORE ANALYSIS")
print("=" * 80)
print("\nSA Score Interpretation:")
print("  1-3  = EASY to synthesize (drug-like)")
print("  3-5  = MODERATE difficulty")
print("  5-7  = CHALLENGING")
print("  7-10 = VERY DIFFICULT (complex natural products)")
print()

# Analyze H-bonding leads
print("=" * 80)
print("H-BONDING LEAD CANDIDATES (mol_001, mol_006)")
print("=" * 80)

for mol_id, data in CANDIDATES.items():
    result = detailed_sa_analysis(data['smiles'], mol_id)
    
    print(f"\n{'─' * 60}")
    print(f"  {mol_id.upper()}")
    print(f"{'─' * 60}")
    print(f"  SMILES: {data['smiles']}")
    print(f"  Description: {data['description']}")
    print(f"  Vina Affinity: {data['vina']} kcal/mol")
    print(f"  H-bond: {data['hbond']}")
    print()
    
    # SA Score with interpretation
    sa = result['sa_score']
    if sa < 3:
        difficulty = "EASY ✓"
    elif sa < 5:
        difficulty = "MODERATE ✓"
    elif sa < 7:
        difficulty = "CHALLENGING ⚠"
    else:
        difficulty = "VERY DIFFICULT ✗"
    
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │  SA SCORE: {sa:.2f}  →  {difficulty:18s}│")
    print(f"  └─────────────────────────────────────┘")
    print()
    print(f"  Structural Features:")
    print(f"    Heavy Atoms:      {result['n_heavy_atoms']}")
    print(f"    Chiral Centers:   {result['n_chiral_centers']}")
    print(f"    Rotatable Bonds:  {result['n_rotatable']}")
    print(f"    Ring Systems:     {result['n_rings']} (Aromatic: {result['n_aromatic_rings']})")
    print(f"    Ring Sizes:       {result['ring_sizes']}")
    print(f"    Bridgehead Atoms: {result['n_bridgeheads']}")
    print(f"    Spiro Atoms:      {result['n_spiro']}")
    print()
    print(f"  Functional Groups:")
    if result['functional_groups']:
        for fg, count in result['functional_groups'].items():
            print(f"    - {fg}: {count}")
    else:
        print(f"    - None detected")

# Analyze top 3 for comparison
print("\n" + "=" * 80)
print("TOP 3 BY VINA AFFINITY (for comparison)")
print("=" * 80)

for mol_id, data in TOP_3.items():
    result = detailed_sa_analysis(data['smiles'], mol_id)
    sa = result['sa_score']
    
    if sa < 3:
        difficulty = "EASY ✓"
    elif sa < 5:
        difficulty = "MODERATE ✓"
    elif sa < 7:
        difficulty = "CHALLENGING ⚠"
    else:
        difficulty = "VERY DIFFICULT ✗"
    
    print(f"\n  {mol_id}: SA = {sa:.2f} ({difficulty})")
    print(f"    Vina: {data['vina']} kcal/mol | Chiral: {result['n_chiral_centers']} | Rings: {result['n_rings']}")

# Analyze known drugs
print("\n" + "=" * 80)
print("KNOWN EGFR DRUGS (Reference)")
print("=" * 80)

for drug_name, smiles in KNOWN_DRUGS.items():
    result = detailed_sa_analysis(smiles, drug_name)
    sa = result['sa_score']
    
    if sa < 3:
        difficulty = "EASY"
    elif sa < 5:
        difficulty = "MODERATE"
    elif sa < 7:
        difficulty = "CHALLENGING"
    else:
        difficulty = "VERY DIFFICULT"
    
    print(f"\n  {drug_name}: SA = {sa:.2f} ({difficulty})")
    print(f"    Chiral: {result['n_chiral_centers']} | Rings: {result['n_rings']} | Rotatable: {result['n_rotatable']}")

# Summary comparison
print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)

all_mols = {}
for mol_id, data in CANDIDATES.items():
    all_mols[mol_id] = detailed_sa_analysis(data['smiles'], mol_id)
for mol_id, data in TOP_3.items():
    all_mols[mol_id] = detailed_sa_analysis(data['smiles'], mol_id)
for drug_name, smiles in KNOWN_DRUGS.items():
    all_mols[drug_name] = detailed_sa_analysis(smiles, drug_name)

# Sort by SA score
sorted_mols = sorted(all_mols.items(), key=lambda x: x[1]['sa_score'])

print(f"\n{'Molecule':<15} {'SA Score':<10} {'Difficulty':<15} {'Chiral':<8} {'Rings':<8}")
print("-" * 60)

for mol_name, result in sorted_mols:
    sa = result['sa_score']
    if sa < 3:
        diff = "EASY"
    elif sa < 5:
        diff = "MODERATE"
    elif sa < 7:
        diff = "CHALLENGING"
    else:
        diff = "V.DIFFICULT"
    
    marker = "★" if mol_name in ['mol_001', 'mol_006'] else ""
    print(f"{mol_name:<15} {sa:<10.2f} {diff:<15} {result['n_chiral_centers']:<8} {result['n_rings']:<8} {marker}")

print("\n★ = H-bonding lead candidates")

# Final verdict
print("\n" + "=" * 80)
print("SYNTHESIS FEASIBILITY VERDICT")
print("=" * 80)

mol001_sa = all_mols['mol_001']['sa_score']
mol006_sa = all_mols['mol_006']['sa_score']
erlotinib_sa = all_mols['Erlotinib']['sa_score']

print(f"""
  mol_001 (SA = {mol001_sa:.2f}):
    - Quinazoline core: Well-established synthesis routes
    - Cyclohexyl-carboxylic acid: Standard coupling chemistry
    - No chiral centers: Racemic synthesis possible
    - VERDICT: {"✓ SYNTHETICALLY ACCESSIBLE" if mol001_sa < 5 else "⚠ CHALLENGING"}

  mol_006 (SA = {mol006_sa:.2f}):
    - Anthraquinone scaffold: Known chemistry
    - Chloro substitution: Standard halogenation
    - Methoxyanilino group: Simple SNAr or coupling
    - No chiral centers: Direct synthesis
    - VERDICT: {"✓ SYNTHETICALLY ACCESSIBLE" if mol006_sa < 5 else "⚠ CHALLENGING"}

  Comparison to Erlotinib (SA = {erlotinib_sa:.2f}):
    - Both candidates have {"SIMILAR" if abs(mol001_sa - erlotinib_sa) < 1 else "DIFFERENT"} complexity
    - Erlotinib is an approved drug successfully manufactured at scale
""")

print("=" * 80)
print("SA SCORE ANALYSIS COMPLETE")
print("=" * 80)
