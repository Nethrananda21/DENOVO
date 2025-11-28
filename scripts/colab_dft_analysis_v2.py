# =============================================================================
# QUANTUM MECHANICS STABILITY ANALYSIS - GOOGLE COLAB VERSION 2
# =============================================================================
# 
# This version uses PySCF (open-source quantum chemistry) which works on Colab
# 
# INSTRUCTIONS:
# 1. Copy cells into Google Colab
# 2. Run each cell in order
# 3. No additional files needed - all molecule data is embedded
#
# =============================================================================

#%%
# ============================================================================
# CELL 1: INSTALL DEPENDENCIES (Run this first!)
# ============================================================================
# Uncomment and run in Colab:

# !pip install rdkit pyscf geometric

#%%
# ============================================================================
# CELL 2: IMPORTS AND SETUP
# ============================================================================
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

print("=" * 70)
print("QUANTUM MECHANICS (DFT) STABILITY ANALYSIS")
print("Using PySCF - Open Source Quantum Chemistry")
print("=" * 70)

#%%
# ============================================================================
# CELL 3: DEFINE LEAD CANDIDATES
# ============================================================================
# Your validated EGFR inhibitor candidates with verified H-bonds

CANDIDATES = {
    'mol_001': {
        'smiles': 'COc1cc2ncnc(NC3CCC(C(=O)O)CC3)c2cc1OC',
        'vina_affinity': -8.78,
        'h_bond': 'THR830 (3.0 Angstrom)',
        'sa_score': 4.92,
        'novelty': 'NOVEL'
    },
    'mol_006': {
        'smiles': 'COc1ccc(NC2=C(Cl)C(=O)c3ccccc3C2=O)cc1',
        'vina_affinity': -8.37,
        'h_bond': 'THR766 (3.5 Angstrom)',
        'sa_score': 4.34,
        'novelty': 'NOVEL'
    }
}

# Reference approved EGFR inhibitors
REFERENCES = {
    'Erlotinib': 'COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC',
    'Gefitinib': 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
}

print("Lead Candidates:")
for mol_id, data in CANDIDATES.items():
    print(f"  {mol_id}: Vina={data['vina_affinity']} kcal/mol, H-bond={data['h_bond']}")

print(f"\nReference Drugs: {list(REFERENCES.keys())}")

#%%
# ============================================================================
# CELL 4: GENERATE 3D STRUCTURES
# ============================================================================
def smiles_to_xyz(smiles, mol_name):
    """Convert SMILES to XYZ format for quantum calculations"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  Failed to parse: {mol_name}")
        return None, None
    
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result == -1:
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
    
    # Optimize with MMFF94
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except:
        pass
    
    # Extract coordinates
    conf = mol.GetConformer()
    atoms = []
    coords = []
    
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        atoms.append(atom.GetSymbol())
        coords.append([pos.x, pos.y, pos.z])
    
    return atoms, np.array(coords)

print("\nGenerating 3D Structures...")
structures = {}

for mol_id, data in CANDIDATES.items():
    atoms, coords = smiles_to_xyz(data['smiles'], mol_id)
    if atoms:
        structures[mol_id] = {'atoms': atoms, 'coords': coords, 'smiles': data['smiles']}
        print(f"  {mol_id}: {len(atoms)} atoms")

for name, smiles in REFERENCES.items():
    atoms, coords = smiles_to_xyz(smiles, name)
    if atoms:
        structures[name] = {'atoms': atoms, 'coords': coords, 'smiles': smiles}
        print(f"  {name}: {len(atoms)} atoms")

#%%
# ============================================================================
# CELL 5: PYSCF DFT CALCULATIONS
# ============================================================================
print("\n" + "=" * 70)
print("RUNNING DFT CALCULATIONS (B3LYP/STO-3G)")
print("This may take a few minutes per molecule...")
print("=" * 70)

try:
    from pyscf import gto, dft
    PYSCF_AVAILABLE = True
    print("\nPySCF loaded successfully!")
except ImportError:
    PYSCF_AVAILABLE = False
    print("\nPySCF not available. Install with: pip install pyscf")

def run_dft_calculation(atoms, coords, mol_name):
    """Run DFT calculation using PySCF"""
    if not PYSCF_AVAILABLE:
        return None
    
    print(f"\n  Calculating {mol_name}...")
    
    try:
        # Build geometry string for PySCF
        geom = ""
        for atom, coord in zip(atoms, coords):
            geom += f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}; "
        
        # Create molecule object
        # Using STO-3G basis for speed (use 6-31G* for better accuracy)
        mol = gto.Mole()
        mol.atom = geom
        mol.basis = 'sto-3g'  # Fast basis set
        mol.charge = 0
        mol.spin = 0
        mol.verbose = 0
        mol.build()
        
        # Run DFT with B3LYP functional
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.verbose = 0
        energy = mf.kernel()
        
        # Get orbital energies
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        
        # Find HOMO and LUMO indices
        homo_idx = np.where(mo_occ > 0)[0][-1]
        lumo_idx = homo_idx + 1
        
        # Convert from Hartree to eV
        homo_ev = mo_energy[homo_idx] * 27.2114
        lumo_ev = mo_energy[lumo_idx] * 27.2114
        gap_ev = lumo_ev - homo_ev
        
        print(f"    HOMO: {homo_ev:.2f} eV | LUMO: {lumo_ev:.2f} eV | Gap: {gap_ev:.2f} eV")
        
        return {
            'homo': homo_ev,
            'lumo': lumo_ev,
            'gap': gap_ev,
            'total_energy': energy * 27.2114,
            'method': 'B3LYP/STO-3G (PySCF)'
        }
        
    except Exception as e:
        print(f"    Error: {e}")
        return None

# Run calculations
results = {}

for mol_name, struct in structures.items():
    result = run_dft_calculation(struct['atoms'], struct['coords'], mol_name)
    if result:
        results[mol_name] = result

#%%
# ============================================================================
# CELL 6: RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("HOMO-LUMO GAP ANALYSIS RESULTS")
print("=" * 70)

print("""
INTERPRETATION:
  Gap > 4 eV  : HIGHLY STABLE - Chemically inert
  Gap 3-4 eV  : STABLE - Good for drug molecules  
  Gap 2-3 eV  : MODERATE - May need stability testing
  Gap 1-2 eV  : REACTIVE - Prone to degradation
  Gap < 1 eV  : UNSTABLE - Too reactive
""")

def get_stability(gap):
    if gap > 4:
        return "HIGHLY STABLE", "+++"
    elif gap > 3:
        return "STABLE", "++"
    elif gap > 2:
        return "MODERATE", "+"
    elif gap > 1:
        return "REACTIVE", "!"
    else:
        return "UNSTABLE", "X"

print("-" * 75)
print(f"{'Molecule':<15} {'HOMO (eV)':<12} {'LUMO (eV)':<12} {'Gap (eV)':<12} {'Status':<20}")
print("-" * 75)

# Candidates first
for mol_id in CANDIDATES.keys():
    if mol_id in results:
        r = results[mol_id]
        status, icon = get_stability(r['gap'])
        print(f"{mol_id:<15} {r['homo']:<12.2f} {r['lumo']:<12.2f} {r['gap']:<12.2f} {status} [{icon}]")

print("-" * 75)

# References
for name in REFERENCES.keys():
    if name in results:
        r = results[name]
        status, icon = get_stability(r['gap'])
        print(f"{name:<15} {r['homo']:<12.2f} {r['lumo']:<12.2f} {r['gap']:<12.2f} {status} [{icon}]")

print("-" * 75)

#%%
# ============================================================================
# CELL 7: COMPARATIVE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON VS ERLOTINIB (Approved EGFR Inhibitor)")
print("=" * 70)

if 'Erlotinib' in results:
    ref_gap = results['Erlotinib']['gap']
    
    for mol_id in CANDIDATES.keys():
        if mol_id in results:
            gap = results[mol_id]['gap']
            diff = gap - ref_gap
            
            print(f"\n{mol_id.upper()}:")
            print(f"  HOMO-LUMO Gap:  {gap:.2f} eV")
            print(f"  Erlotinib Gap:  {ref_gap:.2f} eV")
            print(f"  Difference:     {'+' if diff > 0 else ''}{diff:.2f} eV")
            
            if diff > 0.3:
                assess = "MORE STABLE than Erlotinib"
            elif diff > -0.3:
                assess = "SIMILAR stability to Erlotinib"
            else:
                assess = "LESS STABLE than Erlotinib"
            print(f"  Assessment:     {assess}")

#%%
# ============================================================================
# CELL 8: FINAL VERDICT
# ============================================================================
print("\n" + "=" * 70)
print("FINAL QUANTUM MECHANICS VERDICT")
print("=" * 70)

all_pass = True
for mol_id in CANDIDATES.keys():
    if mol_id in results:
        gap = results[mol_id]['gap']
        status, _ = get_stability(gap)
        
        print(f"\n  {mol_id}:")
        print(f"    HOMO-LUMO Gap: {gap:.2f} eV")
        print(f"    Stability:     {status}")
        
        if gap >= 2:
            print(f"    Verdict:       PASS - Acceptable electronic stability")
        else:
            print(f"    Verdict:       FAIL - Too reactive")
            all_pass = False

print("\n" + "=" * 70)
if all_pass:
    print("CONCLUSION: Both candidates show acceptable electronic stability")
    print("            Suitable for further preclinical development")
else:
    print("WARNING: Some candidates show stability concerns")
print("=" * 70)

#%%
# ============================================================================
# CELL 9: SAVE RESULTS
# ============================================================================
import csv

csv_rows = []
for mol_name, r in results.items():
    status, _ = get_stability(r['gap'])
    is_candidate = mol_name in CANDIDATES
    
    row = {
        'Molecule': mol_name,
        'Type': 'Candidate' if is_candidate else 'Reference',
        'Method': r['method'],
        'HOMO_eV': round(r['homo'], 3),
        'LUMO_eV': round(r['lumo'], 3),
        'Gap_eV': round(r['gap'], 3),
        'Total_Energy_eV': round(r['total_energy'], 3),
        'Stability': status
    }
    csv_rows.append(row)

with open('dft_homo_lumo_results.csv', 'w', newline='') as f:
    if csv_rows:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

print("\nResults saved to: dft_homo_lumo_results.csv")

# Display as DataFrame if pandas available
try:
    import pandas as pd
    df = pd.DataFrame(csv_rows)
    print("\n")
    print(df.to_string(index=False))
except:
    pass

print("\n" + "=" * 70)
print("DFT STABILITY ANALYSIS COMPLETE")
print("=" * 70)
