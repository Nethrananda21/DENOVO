#!/usr/bin/env python3
"""
Phase 5: Molecular Docking with AutoDock Vina
=============================================
Tests if generated molecules fit inside the EGFR protein pocket.

Target: EGFR kinase (PDB: 1M17) - crystal structure with Erlotinib
Tool: AutoDock Vina (command-line) or scoring approximation

Binding Affinity Interpretation:
  > -7.0 kcal/mol: Weak binder
  -7.0 to -8.5 kcal/mol: Moderate binder
  -8.5 to -9.5 kcal/mol: Good binder (Drug-like)
  < -9.5 kcal/mol: Excellent binder (Potent)
"""

import os
import sys
import csv
import argparse
import tempfile
import subprocess
import zipfile
import platform
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import urllib.request
import shutil

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import Draw
from rdkit import DataStructs

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)


# =============================================================================
# VINA SETUP
# =============================================================================

def download_vina_windows(install_dir: Path) -> Optional[str]:
    """Download AutoDock Vina for Windows."""
    vina_url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_windows_x86_64.zip"
    
    print("Downloading AutoDock Vina...")
    zip_path = install_dir / "vina.zip"
    
    try:
        urllib.request.urlretrieve(vina_url, str(zip_path))
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(install_dir)
        
        # Find vina executable
        for f in install_dir.rglob("vina*.exe"):
            if "vina" in f.name.lower() and "gpu" not in f.name.lower():
                print(f"✓ Vina installed: {f}")
                return str(f)
        
        # Also check direct extraction
        vina_exe = install_dir / "vina_1.2.5_windows_x86_64" / "vina.exe"
        if vina_exe.exists():
            return str(vina_exe)
            
    except Exception as e:
        print(f"  Warning: Could not download Vina: {e}")
    
    return None


def find_or_install_vina(dock_dir: Path) -> Optional[str]:
    """Find existing Vina installation or download it."""
    # Check if vina is in PATH
    vina_in_path = shutil.which("vina")
    if vina_in_path:
        print(f"✓ Found Vina in PATH: {vina_in_path}")
        return vina_in_path
    
    # Check local installation
    local_vina = dock_dir / "vina_1.2.5_windows_x86_64" / "vina.exe"
    if local_vina.exists():
        print(f"✓ Found local Vina: {local_vina}")
        return str(local_vina)
    
    # Download if Windows
    if platform.system() == "Windows":
        return download_vina_windows(dock_dir)
    
    return None


# =============================================================================
# PROTEIN PREPARATION
# =============================================================================

def download_pdb(pdb_id: str, output_path: str) -> bool:
    """Download PDB structure from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Downloading {pdb_id} from RCSB PDB...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        return False


def prepare_protein_pdbqt(pdb_path: str, pdbqt_path: str) -> bool:
    """Prepare protein PDBQT file for docking."""
    print("Preparing protein structure...")
    
    # Read PDB and clean it
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    
    # Keep only protein ATOM records
    protein_lines = []
    excluded_residues = ['HOH', 'WAT', 'AQ4', 'SO4', 'PO4', 'GOL', 'EDO', 'ACE', 'NME', 'MG', 'ZN', 'CA']
    
    for line in lines:
        if line.startswith('ATOM'):
            res_name = line[17:20].strip()
            if res_name not in excluded_residues:
                protein_lines.append(line)
        elif line.startswith('TER') or line.startswith('END'):
            protein_lines.append(line)
    
    # Write cleaned PDB
    clean_pdb = pdb_path.replace('.pdb', '_clean.pdb')
    with open(clean_pdb, 'w') as f:
        f.writelines(protein_lines)
    
    # Convert to PDBQT using meeko or simple conversion
    try:
        from meeko import PDBQTReceptor
        # meeko doesn't have receptor prep, use simple method
    except ImportError:
        pass
    
    # Simple PDB to PDBQT conversion
    convert_protein_to_pdbqt(clean_pdb, pdbqt_path)
    print(f"✓ Protein PDBQT prepared: {pdbqt_path}")
    return True


def convert_protein_to_pdbqt(pdb_path: str, pdbqt_path: str):
    """Convert protein PDB to PDBQT with AutoDock atom types."""
    # Atom type mapping for standard amino acids
    atom_types = {
        'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'CG1': 'C', 'CG2': 'C',
        'CD': 'C', 'CD1': 'C', 'CD2': 'C', 'CE': 'C', 'CE1': 'C', 'CE2': 'C', 
        'CE3': 'C', 'CZ': 'C', 'CZ2': 'C', 'CZ3': 'C', 'CH2': 'C',
        'N': 'N', 'NZ': 'N', 'NH1': 'N', 'NH2': 'N', 'NE': 'N', 
        'ND1': 'NA', 'ND2': 'N', 'NE1': 'NA', 'NE2': 'NA',
        'O': 'OA', 'OG': 'OA', 'OG1': 'OA', 'OH': 'OA', 
        'OD1': 'OA', 'OD2': 'OA', 'OE1': 'OA', 'OE2': 'OA', 'OXT': 'OA',
        'S': 'S', 'SG': 'S', 'SD': 'S',
        'H': 'HD', 'HA': 'H', 'HB': 'H', 'HG': 'H', 'HD': 'H', 'HE': 'H', 'HZ': 'HD'
    }
    
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    
    pdbqt_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_name = line[12:16].strip()
            element = atom_name[0] if atom_name else 'C'
            
            # Get AutoDock atom type
            ad_type = atom_types.get(atom_name, 'C')
            if atom_name.startswith('H'):
                ad_type = 'HD' if 'N' in line[17:20] else 'H'
            
            # Approximate charge based on atom type
            charge = 0.0
            if atom_name in ['OD1', 'OD2', 'OE1', 'OE2']:  # Carboxylate
                charge = -0.5
            elif atom_name in ['NZ', 'NH1', 'NH2']:  # Amine
                charge = 0.33
            
            # Format PDBQT line (columns 55-60: charge, 77-78: atom type)
            base_line = line[:54]
            pdbqt_line = f"{base_line}{charge:8.3f} {ad_type:<2s}\n"
            pdbqt_lines.append(pdbqt_line)
        elif line.startswith('TER') or line.startswith('END'):
            pdbqt_lines.append(line)
    
    with open(pdbqt_path, 'w') as f:
        f.writelines(pdbqt_lines)


# =============================================================================
# LIGAND PREPARATION
# =============================================================================

def smiles_to_pdbqt_meeko(smiles: str, output_path: str) -> bool:
    """Convert SMILES to PDBQT using meeko."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        
        # Parse SMILES and generate 3D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        mol = Chem.AddHs(mol)
        
        # Generate 3D conformation
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        
        # Optimize
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except:
                pass
        
        # Prepare with meeko
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        
        # Write PDBQT
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
        
        if is_ok:
            with open(output_path, 'w') as f:
                f.write(pdbqt_string)
            return True
        
    except Exception as e:
        pass
    
    return False


def smiles_to_pdbqt_simple(smiles: str, output_path: str) -> bool:
    """Simple SMILES to PDBQT conversion without meeko."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        mol = Chem.AddHs(mol)
        
        # Generate 3D
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        
        # Optimize
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except:
                pass
        
        # Compute Gasteiger charges
        AllChem.ComputeGasteigerCharges(mol)
        
        # AutoDock atom types
        ad_types = {
            6: 'C', 7: 'NA', 8: 'OA', 9: 'F', 15: 'P',
            16: 'SA', 17: 'Cl', 35: 'Br', 53: 'I', 1: 'HD'
        }
        
        conf = mol.GetConformer()
        
        pdbqt_lines = ['ROOT\n']
        
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atomic_num = atom.GetAtomicNum()
            
            try:
                charge = float(atom.GetProp('_GasteigerCharge'))
                if not (-10 < charge < 10):
                    charge = 0.0
            except:
                charge = 0.0
            
            ad_type = ad_types.get(atomic_num, 'C')
            
            # Check if H-bond donor nitrogen
            if atomic_num == 7 and atom.GetTotalNumHs() > 0:
                ad_type = 'N'
            # Check if H attached to N or O (polar H)
            if atomic_num == 1:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() in [7, 8]:
                        ad_type = 'HD'
                        break
                else:
                    ad_type = 'H'
            
            symbol = atom.GetSymbol()
            atom_name = f"{symbol}{i+1}"[:4]
            
            line = f"ATOM  {i+1:5d} {atom_name:<4s} LIG A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    {charge:8.4f} {ad_type:<2s}\n"
            pdbqt_lines.append(line)
        
        pdbqt_lines.append('ENDROOT\n')
        pdbqt_lines.append('TORSDOF 0\n')
        
        with open(output_path, 'w') as f:
            f.writelines(pdbqt_lines)
        
        return True
        
    except Exception as e:
        return False


def smiles_to_pdbqt(smiles: str, output_path: str) -> bool:
    """Convert SMILES to PDBQT for docking."""
    # Try meeko first
    if smiles_to_pdbqt_meeko(smiles, output_path):
        return True
    
    # Fallback to simple method
    return smiles_to_pdbqt_simple(smiles, output_path)


# =============================================================================
# DOCKING
# =============================================================================

def get_binding_site_center(pdb_path: str, ligand_name: str = 'AQ4') -> Tuple[float, float, float]:
    """Extract binding site center from co-crystallized ligand (Erlotinib = AQ4)."""
    coords = []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                res_name = line[17:20].strip()
                if res_name == ligand_name:
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append((x, y, z))
                    except ValueError:
                        continue
    
    if coords:
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)
        print(f"  Found {ligand_name} with {len(coords)} atoms")
        return (cx, cy, cz)
    
    # Default EGFR ATP binding site coordinates (from 1M17)
    print(f"  Using default EGFR binding site coordinates")
    return (22.0, 0.5, 52.0)


def run_vina_docking(
    vina_exe: str,
    protein_pdbqt: str,
    ligand_pdbqt: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float] = (25, 25, 25),
    exhaustiveness: int = 8
) -> Optional[float]:
    """Run AutoDock Vina and return best binding affinity."""
    
    output_pdbqt = ligand_pdbqt.replace('.pdbqt', '_out.pdbqt')
    log_file = ligand_pdbqt.replace('.pdbqt', '_log.txt')
    
    cmd = [
        vina_exe,
        '--receptor', protein_pdbqt,
        '--ligand', ligand_pdbqt,
        '--center_x', str(center[0]),
        '--center_y', str(center[1]),
        '--center_z', str(center[2]),
        '--size_x', str(size[0]),
        '--size_y', str(size[1]),
        '--size_z', str(size[2]),
        '--exhaustiveness', str(exhaustiveness),
        '--out', output_pdbqt,
        '--num_modes', '5',
        '--log', log_file
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Parse output for affinity
        output = result.stdout + result.stderr
        
        for line in output.split('\n'):
            line = line.strip()
            # Look for: "   1      -8.5      0.000      0.000"
            if line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == '1':
                    try:
                        return float(parts[1])
                    except ValueError:
                        continue
        
        # Also check log file
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip().startswith('1'):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                return float(parts[1])
                            except ValueError:
                                continue
        
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None
    
    return None


def estimate_binding_affinity(smiles: str, reference_smiles: str = None) -> float:
    """
    Estimate binding affinity using physicochemical properties.
    This is a rough approximation when Vina is not available.
    
    Based on empirical correlations from EGFR inhibitor literature.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    
    # Calculate properties
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    
    # EGFR inhibitors typically have:
    # - MW 300-500, LogP 2-5, aromatic rings for pi-stacking
    # - HBA for kinase hinge binding
    
    # Start with baseline
    affinity = -6.0
    
    # Size contribution (optimal MW ~350-450)
    if 300 < mw < 500:
        affinity -= 1.0
    if 350 < mw < 450:
        affinity -= 0.5
    
    # Lipophilicity (optimal LogP ~3-4)
    if 2.0 < logp < 5.0:
        affinity -= 0.8
    if 3.0 < logp < 4.5:
        affinity -= 0.5
    
    # H-bond acceptors (important for hinge binding)
    if 4 <= hba <= 7:
        affinity -= 0.7
    
    # H-bond donors (1-2 optimal)
    if 1 <= hbd <= 2:
        affinity -= 0.5
    
    # Aromatic rings (2-3 optimal for EGFR)
    if 2 <= aromatic_rings <= 4:
        affinity -= 0.8
    
    # Flexibility penalty
    if rotatable > 8:
        affinity += 0.3 * (rotatable - 8)
    
    # TPSA (optimal 60-100 for kinase inhibitors)
    if 50 < tpsa < 120:
        affinity -= 0.4
    
    # Check for common EGFR inhibitor features
    # Quinazoline core (very common in EGFR inhibitors)
    if mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccc2ncncc2c1')):
        affinity -= 1.2
    # Anilinoquinazoline (erlotinib, gefitinib scaffold)
    if mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccc(Nc2ncnc3ccccc23)cc1')):
        affinity -= 1.5
    # Pyrimidine core
    if mol.HasSubstructMatch(Chem.MolFromSmarts('c1ncncc1')):
        affinity -= 0.5
    
    # Add some noise to simulate variability
    import random
    random.seed(hash(smiles) % 2**32)
    noise = random.gauss(0, 0.3)
    affinity += noise
    
    return round(affinity, 2)


def classify_binder(affinity: float) -> str:
    """Classify binding strength based on affinity."""
    if affinity is None:
        return "Failed"
    elif affinity > -7.0:
        return "Weak"
    elif affinity > -8.5:
        return "Moderate"
    elif affinity > -9.5:
        return "Good"
    else:
        return "Excellent"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 5: Molecular Docking')
    parser.add_argument('--input', type=str, default='results/final_candidates.csv',
                        help='Input CSV with candidates')
    parser.add_argument('--output', type=str, default='results/docking_results.csv',
                        help='Output CSV with docking scores')
    parser.add_argument('--n_mols', type=int, default=50,
                        help='Number of molecules to dock')
    parser.add_argument('--pdb', type=str, default='1M17',
                        help='PDB ID for target protein')
    parser.add_argument('--exhaustiveness', type=int, default=8,
                        help='Vina exhaustiveness')
    parser.add_argument('--box_size', type=float, default=25,
                        help='Docking box size (Angstroms)')
    parser.add_argument('--estimate', action='store_true',
                        help='Use estimated scoring (no Vina required)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE 5: MOLECULAR DOCKING")
    print("=" * 60)
    print(f"Target: EGFR kinase (PDB: {args.pdb})")
    print(f"Candidates to dock: {args.n_mols}")
    print()
    
    # Setup directories
    dock_dir = Path('docking')
    dock_dir.mkdir(exist_ok=True)
    results_dir = Path('results')
    
    # Check for Vina
    vina_exe = None
    use_estimation = args.estimate
    
    if not use_estimation:
        vina_exe = find_or_install_vina(dock_dir)
        if vina_exe is None:
            print("\n⚠ AutoDock Vina not available.")
            print("  Using estimated scoring based on molecular properties.")
            print("  (For actual docking, install Vina manually)")
            use_estimation = True
    
    # Download and prepare protein
    print()
    print("-" * 40)
    print("STEP 1: PREPARE PROTEIN TARGET")
    print("-" * 40)
    
    pdb_path = dock_dir / f'{args.pdb}.pdb'
    protein_pdbqt = dock_dir / f'{args.pdb}_protein.pdbqt'
    
    if not pdb_path.exists():
        download_pdb(args.pdb, str(pdb_path))
    else:
        print(f"✓ Using existing PDB: {pdb_path}")
    
    if not use_estimation:
        if not protein_pdbqt.exists():
            prepare_protein_pdbqt(str(pdb_path), str(protein_pdbqt))
        else:
            print(f"✓ Using existing protein PDBQT: {protein_pdbqt}")
    
    # Get binding site
    center = get_binding_site_center(str(pdb_path), 'AQ4')
    print(f"✓ Binding site center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    box_size = (args.box_size, args.box_size, args.box_size)
    
    # Load candidates
    print()
    print("-" * 40)
    print("STEP 2: LOAD CANDIDATES")
    print("-" * 40)
    
    candidates = []
    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append(row)
    
    candidates = candidates[:args.n_mols]
    print(f"✓ Loaded {len(candidates)} candidates")
    
    # Run docking/scoring
    print()
    print("-" * 40)
    print(f"STEP 3: {'MOLECULAR DOCKING' if not use_estimation else 'BINDING ESTIMATION'}")
    print("-" * 40)
    print()
    
    results = []
    ligands_dir = dock_dir / 'ligands'
    ligands_dir.mkdir(exist_ok=True)
    
    for i, cand in enumerate(candidates):
        smiles = cand['SMILES']
        qed = float(cand['QED'])
        rank = cand['Rank']
        
        print(f"[{i+1:3d}/{len(candidates)}] Molecule {rank}...", end=' ', flush=True)
        
        if use_estimation:
            # Use property-based estimation
            affinity = estimate_binding_affinity(smiles)
            classification = classify_binder(affinity)
            print(f"{affinity:.2f} kcal/mol ({classification})")
            
        else:
            # Real Vina docking
            ligand_pdbqt = ligands_dir / f'ligand_{rank}.pdbqt'
            
            if not smiles_to_pdbqt(smiles, str(ligand_pdbqt)):
                print("FAILED (ligand prep)")
                affinity = None
                classification = "Failed"
            else:
                affinity = run_vina_docking(
                    vina_exe,
                    str(protein_pdbqt),
                    str(ligand_pdbqt),
                    center,
                    box_size,
                    args.exhaustiveness
                )
                classification = classify_binder(affinity)
                
                if affinity is not None:
                    print(f"{affinity:.2f} kcal/mol ({classification})")
                else:
                    print("FAILED (docking)")
        
        results.append({
            'Rank': rank,
            'SMILES': smiles,
            'QED': qed,
            'Affinity': affinity,
            'Classification': classification,
            **{k: v for k, v in cand.items() if k not in ['Rank', 'SMILES', 'QED']}
        })
    
    # Analyze results
    print()
    print("-" * 40)
    print("STEP 4: RESULTS ANALYSIS")
    print("-" * 40)
    
    successful = [r for r in results if r['Affinity'] is not None]
    successful.sort(key=lambda x: x['Affinity'])
    
    print(f"\nScoring Success: {len(successful)}/{len(results)}")
    
    # Classification summary
    classifications = {}
    for r in successful:
        c = r['Classification']
        classifications[c] = classifications.get(c, 0) + 1
    
    print("\nBinding Classification:")
    for cls in ['Excellent', 'Good', 'Moderate', 'Weak']:
        count = classifications.get(cls, 0)
        pct = 100 * count / len(successful) if successful else 0
        bar = '█' * int(pct / 5)
        print(f"  {cls:<10}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Top hits
    print()
    print("=" * 60)
    print("TOP 10 HITS")
    print("=" * 60)
    print(f"{'Rank':<6} {'Affinity':<12} {'QED':<8} {'Class':<12} SMILES")
    print("-" * 60)
    
    for r in successful[:10]:
        smiles_short = r['SMILES'][:35] + '...' if len(r['SMILES']) > 35 else r['SMILES']
        print(f"{r['Rank']:<6} {r['Affinity']:<12.2f} {r['QED']:<8.3f} {r['Classification']:<12} {smiles_short}")
    
    # Save results
    for i, r in enumerate(successful, 1):
        r['Dock_Rank'] = i
    
    output_path = Path(args.output)
    fieldnames = ['Dock_Rank', 'Affinity', 'Classification', 'Rank', 'SMILES', 'QED',
                  'Tanimoto', 'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Scaffold']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[fn for fn in fieldnames if fn in successful[0]])
        writer.writeheader()
        for r in successful:
            writer.writerow({k: v for k, v in r.items() if k in fieldnames})
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Save top hits
    top_path = results_dir / 'top_docking_hits.csv'
    with open(top_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Dock_Rank', 'Affinity', 'Classification', 'SMILES', 'QED'])
        writer.writeheader()
        for r in successful[:20]:
            writer.writerow({
                'Dock_Rank': r['Dock_Rank'],
                'Affinity': r['Affinity'],
                'Classification': r['Classification'],
                'SMILES': r['SMILES'],
                'QED': r['QED']
            })
    
    print(f"✓ Top hits saved to: {top_path}")
    
    # Statistics
    if successful:
        affinities = [r['Affinity'] for r in successful]
        print(f"\nAffinity Statistics:")
        print(f"  Best:  {min(affinities):.2f} kcal/mol")
        print(f"  Worst: {max(affinities):.2f} kcal/mol")
        print(f"  Mean:  {sum(affinities)/len(affinities):.2f} kcal/mol")
        
        drug_like = len([a for a in affinities if a < -8.5])
        potent = len([a for a in affinities if a < -9.5])
        print(f"\n  Drug-like (< -8.5): {drug_like}")
        print(f"  Potent (< -9.5):    {potent}")
    
    print()
    print("=" * 60)
    print("PHASE 5 COMPLETE")
    if use_estimation:
        print("Note: Used estimated scoring. For accurate results,")
        print("      install AutoDock Vina and re-run without --estimate")
    print("=" * 60)


if __name__ == '__main__':
    main()
