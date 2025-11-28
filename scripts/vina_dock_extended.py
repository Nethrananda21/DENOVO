#!/usr/bin/env python3
"""
Extended Vina Docking for Remaining 15 Molecules
=================================================
Docks molecules not yet processed: ranks 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
(Already done: 1, 2, 3, 5, 10)
"""

import os
import csv
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile
import shutil

# =============================================================================
# CONFIGURATION
# =============================================================================

VINA_EXE = r"C:\DENOVO\docking\vina_1.2.7_win.exe"
RECEPTOR_PDBQT = r"C:\DENOVO\docking\vina_results\1M17_receptor.pdbqt"
DOCKING_FILE = r"C:\DENOVO\results\top_docking_hits.csv"
OUTPUT_DIR = r"C:\DENOVO\docking\vina_results_extended"

# Binding site center (from Erlotinib in 1M17)
CENTER_X, CENTER_Y, CENTER_Z = 22.0, 0.3, 52.8
BOX_SIZE = 25

# Molecules already docked (ranks from previous run)
ALREADY_DOCKED = {1, 2, 3, 5, 10}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def smiles_to_pdbqt(smiles, output_path):
    """Convert SMILES to PDBQT format"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result != 0:
            return False
    
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Write PDB first
    pdb_path = output_path.replace('.pdbqt', '.pdb')
    Chem.MolToPDBFile(mol, pdb_path)
    
    # Read PDB and add PDBQT headers
    with open(pdb_path, 'r') as f:
        pdb_content = f.read()
    
    # Simple conversion - Vina can work with basic PDBQT
    with open(output_path, 'w') as f:
        f.write("REMARK SMILES: " + smiles + "\n")
        for line in pdb_content.split('\n'):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Extend line to PDBQT format (add charge and type columns)
                atom_name = line[12:16].strip()
                # Determine atom type from name
                if atom_name.startswith('C'):
                    atom_type = 'C'
                elif atom_name.startswith('N'):
                    atom_type = 'N'
                elif atom_name.startswith('O'):
                    atom_type = 'OA'
                elif atom_name.startswith('S'):
                    atom_type = 'SA'
                elif atom_name.startswith('F'):
                    atom_type = 'F'
                elif atom_name.startswith('Cl') or atom_name.startswith('CL'):
                    atom_type = 'Cl'
                elif atom_name.startswith('Br') or atom_name.startswith('BR'):
                    atom_type = 'Br'
                elif atom_name.startswith('H'):
                    atom_type = 'HD' if 'N' in line or 'O' in line else 'H'
                else:
                    atom_type = 'C'
                
                # Format PDBQT line
                pdbqt_line = line[:54].ljust(54) + "  0.00" + f"  {atom_type:>2}"
                f.write(pdbqt_line + "\n")
            elif line.startswith('END'):
                f.write(line + "\n")
    
    return True

def run_vina(ligand_pdbqt, output_pdbqt, log_file):
    """Run AutoDock Vina"""
    cmd = [
        VINA_EXE,
        '--receptor', RECEPTOR_PDBQT,
        '--ligand', ligand_pdbqt,
        '--out', output_pdbqt,
        '--center_x', str(CENTER_X),
        '--center_y', str(CENTER_Y),
        '--center_z', str(CENTER_Z),
        '--size_x', str(BOX_SIZE),
        '--size_y', str(BOX_SIZE),
        '--size_z', str(BOX_SIZE),
        '--exhaustiveness', '8',
        '--num_modes', '5'
    ]
    
    with open(log_file, 'w') as lf:
        result = subprocess.run(cmd, capture_output=True, text=True)
        lf.write(result.stdout)
        lf.write(result.stderr)
    
    return result.returncode == 0

def parse_vina_output(pdbqt_file):
    """Extract best binding affinity from Vina output"""
    if not os.path.exists(pdbqt_file):
        return None
    
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if 'VINA RESULT' in line or 'REMARK VINA RESULT' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    try:
                        val = float(p)
                        return val
                    except ValueError:
                        continue
    return None

# =============================================================================
# SETUP DIRECTORIES
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'ligands'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'poses'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'logs'), exist_ok=True)

print("=" * 80)
print("EXTENDED VINA DOCKING - REMAINING 15 MOLECULES")
print("=" * 80)

# =============================================================================
# LOAD MOLECULES
# =============================================================================

molecules = []
with open(DOCKING_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rank = int(row['Dock_Rank'])
        if rank not in ALREADY_DOCKED:
            molecules.append({
                'rank': rank,
                'smiles': row['SMILES'],
                'estimated_affinity': float(row['Affinity'])
            })

print(f"\nMolecules to dock: {len(molecules)}")
print(f"Already docked (skipping): {sorted(ALREADY_DOCKED)}")

# =============================================================================
# RUN DOCKING
# =============================================================================

results = []

for mol_data in molecules:
    rank = mol_data['rank']
    smiles = mol_data['smiles']
    estimated = mol_data['estimated_affinity']
    
    mol_id = f"mol_{rank:03d}"
    print(f"\n[{mol_id}] {smiles[:50]}...")
    
    # File paths
    ligand_pdbqt = os.path.join(OUTPUT_DIR, 'ligands', f'{mol_id}.pdbqt')
    output_pdbqt = os.path.join(OUTPUT_DIR, 'poses', f'{mol_id}_docked.pdbqt')
    log_file = os.path.join(OUTPUT_DIR, 'logs', f'{mol_id}.log')
    
    # Prepare ligand
    print(f"  Preparing ligand...", end=' ')
    if not smiles_to_pdbqt(smiles, ligand_pdbqt):
        print("FAILED (could not generate 3D)")
        results.append({'rank': rank, 'smiles': smiles, 'estimated': estimated, 'vina': None, 'status': 'prep_failed'})
        continue
    print("OK")
    
    # Run Vina
    print(f"  Running Vina...", end=' ')
    if not run_vina(ligand_pdbqt, output_pdbqt, log_file):
        print("FAILED (Vina error)")
        results.append({'rank': rank, 'smiles': smiles, 'estimated': estimated, 'vina': None, 'status': 'dock_failed'})
        continue
    
    # Parse results
    affinity = parse_vina_output(output_pdbqt)
    if affinity is not None:
        print(f"OK → {affinity:.2f} kcal/mol (estimated: {estimated:.2f})")
        results.append({'rank': rank, 'smiles': smiles, 'estimated': estimated, 'vina': affinity, 'status': 'success'})
    else:
        print("FAILED (no result)")
        results.append({'rank': rank, 'smiles': smiles, 'estimated': estimated, 'vina': None, 'status': 'parse_failed'})

# =============================================================================
# SAVE RESULTS
# =============================================================================

results_file = os.path.join(OUTPUT_DIR, 'vina_extended_results.csv')
with open(results_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['rank', 'smiles', 'estimated', 'vina', 'status'])
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 80)
print("DOCKING COMPLETE")
print("=" * 80)

# Summary
success_count = sum(1 for r in results if r['status'] == 'success')
print(f"\nSuccessful: {success_count}/{len(molecules)}")

# Sort by Vina affinity
successful = [r for r in results if r['vina'] is not None]
successful.sort(key=lambda x: x['vina'])

print(f"\n{'Rank':<8}{'Estimated':<14}{'Vina Actual':<14}{'Status'}")
print("-" * 50)
for r in successful:
    print(f"{r['rank']:<8}{r['estimated']:<14.2f}{r['vina']:<14.2f}{r['status']}")

print(f"\nResults saved to: {results_file}")
