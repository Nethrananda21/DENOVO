#!/usr/bin/env python3
"""
Stage 3: Uni-Mol Transformer-based 3D Structure Generation
==========================================================
Method: Uni-Mol - BERT-like Transformer for Molecular Conformations

Why Uni-Mol?
------------
Uni-Mol uses a Transformer architecture (similar to BERT) trained on 209 million 
molecular conformations. Unlike physics-based methods (RDKit, ANI-2x), it learns 
the statistical distribution of molecular shapes from data.

Key Advantages:
- Captures molecular flexibility and "fuzziness"
- Generates diverse, chemically valid conformers
- Learned from massive conformational datasets
- Better for flexible/drug-like molecules
- Can generate ensemble of conformers for entropy estimation

Architecture:
- 3D Transformer with SE(3)-equivariant attention
- Trained on PCQM4Mv2 + PubChem3D conformations
- Predicts atomic positions directly from 2D graph

Reference:
- Uni-Mol: https://github.com/dptech-corp/Uni-Mol
- Paper: Zhou et al., ICLR 2023
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# RDKit for molecule handling
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

# Check for Uni-Mol
UNIMOL_AVAILABLE = False
try:
    from unimol_tools import UniMolRepr, MolPredict
    UNIMOL_AVAILABLE = True
except ImportError:
    pass

# Alternative: Try unimol from huggingface
UNIMOL_HF_AVAILABLE = False
try:
    from transformers import AutoModel, AutoTokenizer
    UNIMOL_HF_AVAILABLE = True
except ImportError:
    pass


def check_unimol_installation() -> List[str]:
    """Check Uni-Mol installation status."""
    issues = []
    
    if not UNIMOL_AVAILABLE:
        issues.append("unimol_tools not installed (pip install unimol_tools)")
    
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA not available (Uni-Mol benefits from GPU)")
    except ImportError:
        issues.append("PyTorch not installed")
    
    return issues


def generate_conformers_unimol(
    smiles: str,
    num_conformers: int = 10,
    use_gpu: bool = True
) -> Optional[List[Chem.Mol]]:
    """
    Generate conformer ensemble using Uni-Mol.
    
    Args:
        smiles: Input SMILES string
        num_conformers: Number of conformers to generate
        use_gpu: Whether to use GPU
    
    Returns:
        List of RDKit Mol objects with 3D coordinates
    """
    if not UNIMOL_AVAILABLE:
        return None
    
    try:
        # Initialize Uni-Mol predictor
        predictor = MolPredict(
            task='conf_gen',
            use_gpu=use_gpu
        )
        
        # Generate conformers
        results = predictor.predict(
            smiles=[smiles],
            num_confs=num_conformers
        )
        
        conformers = []
        for conf_data in results:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            
            # Set coordinates from Uni-Mol output
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i, coord in enumerate(conf_data['coordinates']):
                conf.SetAtomPosition(i, coord)
            
            mol.AddConformer(conf, assignId=True)
            conformers.append(mol)
        
        return conformers
        
    except Exception as e:
        print(f"Uni-Mol error: {e}")
        return None


def generate_diverse_conformers_rdkit(
    smiles: str,
    num_conformers: int = 10,
    prune_rms_thresh: float = 0.5
) -> Optional[Chem.Mol]:
    """
    Generate diverse conformer ensemble using RDKit ETKDG with pruning.
    Fallback when Uni-Mol is not available.
    
    This mimics Uni-Mol's diversity by:
    1. Generating many conformers
    2. Clustering/pruning by RMSD
    3. Keeping diverse representatives
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        
        # ETKDG parameters for diversity
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 0
        params.pruneRmsThresh = prune_rms_thresh  # RMSD threshold for diversity
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        
        # Generate more conformers than needed, then prune
        n_attempts = num_conformers * 5
        
        conf_ids = AllChem.EmbedMultipleConfs(
            mol, 
            numConfs=n_attempts,
            params=params
        )
        
        if len(conf_ids) == 0:
            # Fallback to random coords
            params.useRandomCoords = True
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
        
        if len(conf_ids) == 0:
            return None
        
        # Optimize all conformers
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500, numThreads=0)
        
        # Get energies and sort
        energies = [(conf_id, res[1]) for conf_id, res in zip(conf_ids, results) if res[0] == 0]
        energies.sort(key=lambda x: x[1])
        
        # Keep only the requested number of diverse conformers
        if len(energies) > num_conformers:
            # Remove excess conformers (keep lowest energy diverse set)
            conf_ids_to_keep = set([e[0] for e in energies[:num_conformers]])
            conf_ids_to_remove = [cid for cid in conf_ids if cid not in conf_ids_to_keep]
            for cid in sorted(conf_ids_to_remove, reverse=True):
                mol.RemoveConformer(cid)
        
        return mol
        
    except Exception as e:
        print(f"RDKit conformer generation error: {e}")
        return None


def calculate_conformer_properties(mol: Chem.Mol) -> Dict:
    """Calculate properties for conformer ensemble."""
    props = {
        'num_conformers': mol.GetNumConformers(),
        'num_atoms': mol.GetNumAtoms(),
        'num_rotatable': rdMolDescriptors.CalcNumRotatableBonds(Chem.RemoveHs(mol))
    }
    
    if mol.GetNumConformers() > 0:
        # Calculate energies for all conformers
        energies = []
        for conf in mol.GetConformers():
            try:
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol, 
                    AllChem.MMFFGetMoleculeProperties(mol),
                    confId=conf.GetId()
                )
                if ff:
                    energies.append(ff.CalcEnergy())
            except:
                pass
        
        if energies:
            props['min_energy'] = min(energies)
            props['max_energy'] = max(energies)
            props['energy_range'] = max(energies) - min(energies)
            props['mean_energy'] = sum(energies) / len(energies)
        
        # Calculate RMSD between conformers (measure of diversity)
        if mol.GetNumConformers() > 1:
            from rdkit.Chem import rdMolAlign
            rmsds = []
            conf_ids = [c.GetId() for c in mol.GetConformers()]
            ref_id = conf_ids[0]
            
            for cid in conf_ids[1:]:
                try:
                    rmsd = rdMolAlign.GetBestRMS(mol, mol, ref_id, cid)
                    rmsds.append(rmsd)
                except:
                    pass
            
            if rmsds:
                props['mean_rmsd'] = sum(rmsds) / len(rmsds)
                props['max_rmsd'] = max(rmsds)
    
    return props


def save_conformer_ensemble(
    mol: Chem.Mol,
    name: str,
    output_dir: Path,
    smiles: str,
    extra_props: Dict = None
):
    """Save conformer ensemble to files."""
    
    # Set molecule properties
    mol.SetProp('_Name', name)
    mol.SetProp('SMILES', smiles)
    
    if extra_props:
        for key, value in extra_props.items():
            if value is not None:
                mol.SetProp(str(key), str(value))
    
    # Save all conformers to single SDF (multi-conformer)
    sdf_dir = output_dir / 'sdf'
    sdf_dir.mkdir(exist_ok=True)
    
    sdf_path = sdf_dir / f'{name}.sdf'
    writer = Chem.SDWriter(str(sdf_path))
    
    for conf in mol.GetConformers():
        writer.write(mol, confId=conf.GetId())
    
    writer.close()
    
    # Save lowest energy conformer as PDB
    pdb_dir = output_dir / 'pdb'
    pdb_dir.mkdir(exist_ok=True)
    
    if mol.GetNumConformers() > 0:
        # Find lowest energy conformer
        best_conf_id = 0
        best_energy = float('inf')
        
        for conf in mol.GetConformers():
            try:
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol,
                    AllChem.MMFFGetMoleculeProperties(mol),
                    confId=conf.GetId()
                )
                if ff:
                    energy = ff.CalcEnergy()
                    if energy < best_energy:
                        best_energy = energy
                        best_conf_id = conf.GetId()
            except:
                pass
        
        pdb_path = pdb_dir / f'{name}.pdb'
        Chem.MolToPDBFile(mol, str(pdb_path), confId=best_conf_id)
    
    return sdf_path


def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: Uni-Mol Transformer-based Conformer Generation'
    )
    parser.add_argument('--input', type=str, default='../top_docking_hits.csv',
                        help='Input CSV with SMILES')
    parser.add_argument('--output_dir', type=str, default='stage3_unimol_ensemble',
                        help='Output directory')
    parser.add_argument('--n_mols', type=int, default=20,
                        help='Number of molecules to process')
    parser.add_argument('--num_conformers', type=int, default=10,
                        help='Number of conformers per molecule')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Disable GPU')
    args = parser.parse_args()
    
    print("=" * 60)
    print("STAGE 3: CONFORMER ENSEMBLE GENERATION")
    print("Method: Uni-Mol Transformer / RDKit Diverse ETKDG")
    print("=" * 60)
    print()
    
    # Check installation
    print("Checking dependencies...")
    issues = check_unimol_installation()
    
    use_unimol = UNIMOL_AVAILABLE and len([i for i in issues if 'unimol' in i.lower()]) == 0
    
    if issues:
        print("\nDependency Status:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    
    if use_unimol:
        print("\n✓ Using Uni-Mol Transformer for conformer generation")
    else:
        print("\n✓ Using RDKit Diverse ETKDG (Uni-Mol-like diversity)")
        print("  (For true Uni-Mol, install: pip install unimol_tools)")
    
    print()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load molecules
    print(f"Loading molecules from: {args.input}")
    molecules = []
    
    input_path = Path(args.input)
    if not input_path.exists():
        input_path = Path(__file__).parent / args.input
    
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            molecules.append(row)
    
    molecules = molecules[:args.n_mols]
    print(f"Processing {len(molecules)} molecules")
    print(f"Conformers per molecule: {args.num_conformers}")
    print()
    
    print("-" * 60)
    
    results = []
    total_conformers = 0
    
    for i, mol_data in enumerate(molecules):
        smiles = mol_data['SMILES']
        rank = int(mol_data['Dock_Rank'])
        name = f"mol_{rank:03d}"
        
        print(f"[{i+1:3d}/{len(molecules)}] Rank {rank}: ", end='', flush=True)
        
        # Generate conformers
        if use_unimol:
            conformers = generate_conformers_unimol(
                smiles,
                num_conformers=args.num_conformers,
                use_gpu=not args.no_gpu
            )
            method = "Uni-Mol"
            
            if conformers is None:
                # Fallback to RDKit
                mol = generate_diverse_conformers_rdkit(smiles, args.num_conformers)
                method = "RDKit-Diverse (fallback)"
            else:
                # Combine conformers into single mol object
                mol = conformers[0] if conformers else None
        else:
            mol = generate_diverse_conformers_rdkit(smiles, args.num_conformers)
            method = "RDKit-Diverse"
        
        if mol is None or mol.GetNumConformers() == 0:
            print("FAILED")
            results.append({
                'Dock_Rank': rank,
                'SMILES': smiles,
                'Method': method,
                'Status': 'Failed'
            })
            continue
        
        # Calculate properties
        props = calculate_conformer_properties(mol)
        
        # Save ensemble
        extra_props = {
            'Affinity': mol_data.get('Affinity', ''),
            'QED': mol_data.get('QED', ''),
            'Method': method
        }
        
        save_conformer_ensemble(mol, name, output_dir, smiles, extra_props)
        
        n_confs = mol.GetNumConformers()
        total_conformers += n_confs
        
        # Display results
        energy_str = f"ΔE={props.get('energy_range', 0):.1f}" if 'energy_range' in props else ""
        rmsd_str = f"RMSD={props.get('mean_rmsd', 0):.2f}Å" if 'mean_rmsd' in props else ""
        
        print(f"✓ {n_confs} conformers, {energy_str} kcal/mol, {rmsd_str}")
        
        results.append({
            'Dock_Rank': rank,
            'SMILES': smiles,
            'Num_Conformers': n_confs,
            'Num_Rotatable': props.get('num_rotatable', 0),
            'Min_Energy': props.get('min_energy', ''),
            'Max_Energy': props.get('max_energy', ''),
            'Energy_Range': props.get('energy_range', ''),
            'Mean_RMSD': props.get('mean_rmsd', ''),
            'Max_RMSD': props.get('max_rmsd', ''),
            'Method': method,
            'Status': 'Success'
        })
    
    # Save summary
    print()
    print("-" * 60)
    
    summary_path = output_dir / 'ensemble_summary.csv'
    fieldnames = list(results[0].keys()) if results else []
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Summary saved: {summary_path}")
    
    # Create combined ensemble file
    sdf_dir = output_dir / 'sdf'
    if sdf_dir.exists():
        combined_sdf = output_dir / 'all_ensembles.sdf'
        writer = Chem.SDWriter(str(combined_sdf))
        
        for sdf_file in sorted(sdf_dir.glob('*.sdf')):
            suppl = Chem.SDMolSupplier(str(sdf_file))
            for mol in suppl:
                if mol is not None:
                    writer.write(mol)
        writer.close()
        print(f"✓ Combined ensembles: {combined_sdf}")
    
    # Statistics
    successful = [r for r in results if r['Status'] == 'Success']
    
    print()
    print("=" * 60)
    print("STAGE 3 COMPLETE")
    print("=" * 60)
    print(f"\nResults Summary:")
    print(f"  Molecules processed: {len(successful)}/{len(molecules)}")
    print(f"  Total conformers: {total_conformers}")
    print(f"  Avg conformers/mol: {total_conformers/len(successful):.1f}" if successful else "")
    
    # Flexibility analysis
    if successful:
        avg_rotatable = sum(r.get('Num_Rotatable', 0) for r in successful) / len(successful)
        avg_rmsd = sum(float(r.get('Mean_RMSD', 0) or 0) for r in successful) / len(successful)
        avg_energy_range = sum(float(r.get('Energy_Range', 0) or 0) for r in successful) / len(successful)
        
        print(f"\nFlexibility Analysis:")
        print(f"  Avg rotatable bonds: {avg_rotatable:.1f}")
        print(f"  Avg conformer RMSD: {avg_rmsd:.2f} Å")
        print(f"  Avg energy range: {avg_energy_range:.1f} kcal/mol")
    
    print()
    print("Conformer ensembles are useful for:")
    print("  • Flexible docking (accounting for ligand flexibility)")
    print("  • Entropy estimation (conformational entropy)")
    print("  • Pharmacophore screening (multiple conformations)")
    print("  • Free energy calculations")


if __name__ == '__main__':
    main()
