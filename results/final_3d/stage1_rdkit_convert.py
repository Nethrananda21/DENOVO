#!/usr/bin/env python3
"""
Convert 2D SMILES to 3D Molecular Structures
=============================================
Method 1: RDKit ETKDGv3 - The "Standard & Reliable" Option

ETKDGv3 (Experimental-Torsion Knowledge Distance Geometry v3) is:
- Based on experimental torsion angle preferences from Cambridge Structural Database
- Uses distance geometry for initial embedding
- Includes small ring handling and macrocycle conformer sampling
- Most reliable for drug-like molecules

Output formats:
- SDF (Structure Data File) - standard for 3D structures
- PDB (Protein Data Bank) - for docking software
- MOL2 (Tripos format) - for some visualization tools
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdmolfiles import MolToPDBFile, MolToMolFile

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)


def smiles_to_3d_etkdgv3(
    smiles: str,
    num_conformers: int = 1,
    optimize: bool = True,
    random_seed: int = 42
) -> Optional[Chem.Mol]:
    """
    Convert SMILES to 3D structure using ETKDGv3.
    
    Args:
        smiles: Input SMILES string
        num_conformers: Number of conformers to generate (best one kept)
        optimize: Whether to optimize geometry with force field
        random_seed: Random seed for reproducibility
    
    Returns:
        RDKit Mol object with 3D coordinates, or None if failed
    """
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add hydrogens (essential for realistic 3D geometry)
    mol = Chem.AddHs(mol)
    
    # Setup ETKDGv3 parameters
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0  # Use all available threads
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.enforceChirality = True
    
    # Generate conformer(s)
    if num_conformers > 1:
        # Generate multiple conformers and keep the best
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
        if len(conf_ids) == 0:
            # Fallback to random coordinates
            params.useRandomCoords = True
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
        
        if len(conf_ids) == 0:
            return None
        
        # Optimize and find best conformer
        if optimize:
            energies = []
            for conf_id in conf_ids:
                try:
                    # Try MMFF first (better for drug-like molecules)
                    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id)
                    if ff is not None:
                        ff.Minimize(maxIts=500)
                        energies.append((conf_id, ff.CalcEnergy()))
                    else:
                        # Fallback to UFF
                        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
                        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                        energies.append((conf_id, ff.CalcEnergy() if ff else float('inf')))
                except:
                    energies.append((conf_id, float('inf')))
            
            # Keep only the lowest energy conformer
            if energies:
                best_conf_id = min(energies, key=lambda x: x[1])[0]
                # Remove other conformers
                conf_ids_to_remove = [cid for cid in conf_ids if cid != best_conf_id]
                for cid in sorted(conf_ids_to_remove, reverse=True):
                    mol.RemoveConformer(cid)
    else:
        # Single conformer
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            # Fallback to random coordinates
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol, params)
        
        if result == -1:
            return None
        
        # Optimize geometry
        if optimize:
            try:
                # Try MMFF force field (better for drug molecules)
                mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
                if mmff_props is not None:
                    AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s', maxIters=500)
                else:
                    # Fallback to UFF
                    AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            except Exception as e:
                # If optimization fails, keep unoptimized structure
                pass
    
    return mol


def calculate_3d_properties(mol: Chem.Mol) -> Dict:
    """Calculate 3D molecular properties."""
    props = {}
    
    try:
        # Basic properties
        props['num_atoms'] = mol.GetNumAtoms()
        props['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
        props['num_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        
        # 3D properties (if conformer exists)
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            
            # Calculate radius of gyration
            coords = conf.GetPositions()
            centroid = coords.mean(axis=0)
            distances = ((coords - centroid) ** 2).sum(axis=1)
            props['radius_of_gyration'] = (distances.mean()) ** 0.5
            
            # Asphericity (shape descriptor) - approximate
            from numpy import linalg
            centered = coords - centroid
            inertia = centered.T @ centered / len(coords)
            eigenvalues = sorted(linalg.eigvalsh(inertia), reverse=True)
            if eigenvalues[0] > 0:
                props['asphericity'] = eigenvalues[0] - 0.5 * (eigenvalues[1] + eigenvalues[2])
            
        # Calculate MMFF energy if possible
        try:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
            if mmff_props:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
                if ff:
                    props['mmff_energy'] = ff.CalcEnergy()
        except:
            pass
            
    except Exception as e:
        pass
    
    return props


def main():
    parser = argparse.ArgumentParser(description='Convert SMILES to 3D structures using ETKDGv3')
    parser.add_argument('--input', type=str, default='../top_docking_hits.csv',
                        help='Input CSV with SMILES')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for 3D structures')
    parser.add_argument('--n_mols', type=int, default=20,
                        help='Number of molecules to convert')
    parser.add_argument('--num_conformers', type=int, default=10,
                        help='Number of conformers to generate (best kept)')
    parser.add_argument('--format', type=str, default='all',
                        choices=['sdf', 'pdb', 'mol', 'all'],
                        help='Output format')
    parser.add_argument('--no_optimize', action='store_true',
                        help='Skip geometry optimization')
    args = parser.parse_args()
    
    print("=" * 60)
    print("2D SMILES → 3D STRUCTURE CONVERSION")
    print("Method: RDKit ETKDGv3 (Standard & Reliable)")
    print("=" * 60)
    print()
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    sdf_dir = output_dir / 'sdf'
    pdb_dir = output_dir / 'pdb'
    mol_dir = output_dir / 'mol'
    
    for d in [sdf_dir, pdb_dir, mol_dir]:
        d.mkdir(exist_ok=True)
    
    # Load molecules
    print(f"Loading molecules from: {args.input}")
    molecules = []
    
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = output_dir / args.input
    
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            molecules.append(row)
    
    molecules = molecules[:args.n_mols]
    print(f"Processing {len(molecules)} molecules")
    print(f"Conformers to generate: {args.num_conformers} (best kept)")
    print(f"Geometry optimization: {'Yes (MMFF94s)' if not args.no_optimize else 'No'}")
    print()
    
    # Process each molecule
    print("-" * 60)
    results = []
    successful = 0
    failed = 0
    
    for i, mol_data in enumerate(molecules):
        smiles = mol_data['SMILES']
        rank = int(mol_data.get('Dock_Rank', i+1))
        affinity = mol_data.get('Affinity', 'N/A')
        qed = mol_data.get('QED', 'N/A')
        
        print(f"[{i+1:3d}/{len(molecules)}] Rank {rank}: ", end='', flush=True)
        
        # Convert to 3D
        mol = smiles_to_3d_etkdgv3(
            smiles,
            num_conformers=args.num_conformers,
            optimize=not args.no_optimize
        )
        
        if mol is None:
            print("FAILED (embedding)")
            failed += 1
            continue
        
        # Set molecule name and properties
        mol.SetProp('_Name', f'DENOVO_EGFR_{rank}')
        mol.SetProp('SMILES', smiles)
        mol.SetProp('Dock_Rank', str(rank))
        mol.SetProp('Affinity', str(affinity))
        mol.SetProp('QED', str(qed))
        
        # Calculate 3D properties
        props_3d = calculate_3d_properties(mol)
        
        # Save in requested formats
        base_name = f'mol_{rank:03d}'
        
        try:
            if args.format in ['sdf', 'all']:
                sdf_path = sdf_dir / f'{base_name}.sdf'
                writer = Chem.SDWriter(str(sdf_path))
                writer.write(mol)
                writer.close()
            
            if args.format in ['pdb', 'all']:
                pdb_path = pdb_dir / f'{base_name}.pdb'
                MolToPDBFile(mol, str(pdb_path))
            
            if args.format in ['mol', 'all']:
                mol_path = mol_dir / f'{base_name}.mol'
                MolToMolFile(mol, str(mol_path))
            
            successful += 1
            energy_str = f", E={props_3d.get('mmff_energy', 'N/A'):.1f}" if 'mmff_energy' in props_3d else ""
            print(f"✓ {mol.GetNumAtoms()} atoms{energy_str}")
            
            results.append({
                'Dock_Rank': rank,
                'SMILES': smiles,
                'Affinity': affinity,
                'QED': qed,
                'Num_Atoms': mol.GetNumAtoms(),
                'MMFF_Energy': props_3d.get('mmff_energy', ''),
                'Radius_Gyration': props_3d.get('radius_of_gyration', ''),
                'Status': 'Success'
            })
            
        except Exception as e:
            print(f"FAILED (save): {e}")
            failed += 1
    
    # Create combined SDF file
    print()
    print("-" * 60)
    print("Creating combined structure file...")
    
    combined_sdf = output_dir / 'all_molecules.sdf'
    writer = Chem.SDWriter(str(combined_sdf))
    
    for sdf_file in sorted(sdf_dir.glob('*.sdf')):
        suppl = Chem.SDMolSupplier(str(sdf_file))
        for mol in suppl:
            if mol is not None:
                writer.write(mol)
    writer.close()
    print(f"✓ Combined SDF: {combined_sdf}")
    
    # Save results summary
    summary_path = output_dir / 'conversion_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        fieldnames = ['Dock_Rank', 'SMILES', 'Affinity', 'QED', 'Num_Atoms', 'MMFF_Energy', 'Radius_Gyration', 'Status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Summary: {summary_path}")
    
    # Final report
    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Successful: {successful}/{len(molecules)} ({100*successful/len(molecules):.1f}%)")
    print(f"  Failed:     {failed}/{len(molecules)}")
    print()
    print("Output files:")
    print(f"  SDF files:  {sdf_dir}/")
    print(f"  PDB files:  {pdb_dir}/")
    print(f"  MOL files:  {mol_dir}/")
    print(f"  Combined:   {combined_sdf}")
    print()
    print("These 3D structures can be used for:")
    print("  - Molecular docking (AutoDock Vina, GOLD, Glide)")
    print("  - Visualization (PyMOL, VMD, Chimera)")
    print("  - MD simulations (GROMACS, AMBER)")
    print("  - Pharmacophore modeling")


if __name__ == '__main__':
    main()
