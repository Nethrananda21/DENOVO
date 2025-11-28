#!/usr/bin/env python3
"""
Stage 2: High-Precision 3D Structure Generation with Auto3D
============================================================
Method: Auto3D - The Physics-AI Hybrid

Why Auto3D?
-----------
While RDKit uses a classical "ball and spring" model (MMFF94s), Auto3D uses 
Neural Network Potentials (ANI-2x or AIMNet2) trained on high-level Quantum 
Mechanics (DFT) data.

Result: Bond lengths and angles that match quantum reality, which is crucial 
for accurate docking scores and binding affinity predictions.

Pipeline:
1. Initial conformer generation (ETKDG or similar)
2. Conformer ranking with ANI-2x neural network potential
3. Geometry optimization with quantum-accurate forces
4. Final selection of lowest energy conformer

Reference:
- Auto3D: https://github.com/isayevlab/Auto3D_pkg
- ANI-2x: https://doi.org/10.1021/acs.jctc.0c00121
"""

import os
import sys
import csv
import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Dict
import tempfile

# Check for Auto3D
try:
    from Auto3D.auto3D import options, main as auto3d_main
    AUTO3D_AVAILABLE = True
except ImportError:
    AUTO3D_AVAILABLE = False

# RDKit for fallback and validation
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# Suppress warnings
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)


def check_auto3d_installation():
    """Check if Auto3D and its dependencies are properly installed."""
    issues = []
    
    if not AUTO3D_AVAILABLE:
        issues.append("Auto3D package not found")
    
    # Check for torch
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA not available (Auto3D works best with GPU)")
    except ImportError:
        issues.append("PyTorch not installed")
    
    # Check for torchani (ANI neural network)
    try:
        import torchani
    except ImportError:
        issues.append("torchani not installed (required for ANI-2x potential)")
    
    return issues


def prepare_input_smi(molecules: List[Dict], output_path: str):
    """Prepare input SMI file for Auto3D."""
    with open(output_path, 'w') as f:
        for mol in molecules:
            smiles = mol['SMILES']
            name = f"DENOVO_EGFR_{mol['Dock_Rank']}"
            f.write(f"{smiles} {name}\n")
    return output_path


def run_auto3d_optimization(
    input_smi: str,
    output_dir: str,
    k: int = 1,
    optimizing_engine: str = "ANI2x",
    use_gpu: bool = True,
    max_confs: int = 10
) -> Optional[str]:
    """
    Run Auto3D for high-precision 3D structure generation.
    
    Args:
        input_smi: Path to input SMI file
        output_dir: Directory for output files
        k: Number of conformers to keep per molecule
        optimizing_engine: 'ANI2x', 'ANI2xt', or 'AIMNET'
        use_gpu: Whether to use GPU acceleration
        max_confs: Maximum conformers to generate before ranking
    
    Returns:
        Path to output SDF file, or None if failed
    """
    if not AUTO3D_AVAILABLE:
        return None
    
    try:
        # Setup Auto3D options
        args = options(
            input_smi,
            k=k,
            optimizing_engine=optimizing_engine,
            use_gpu=use_gpu,
            capacity=max_confs,
            enumerate_tautomer=False,
            enumerate_isomer=False,
            verbose=True
        )
        
        # Run Auto3D
        output_path = auto3d_main(args)
        
        return output_path
        
    except Exception as e:
        print(f"Auto3D error: {e}")
        return None


def fallback_ani_optimization(smiles: str, name: str) -> Optional[Chem.Mol]:
    """
    Fallback: Use torchani directly for ANI-2x optimization.
    This is used when Auto3D is not available but torchani is.
    """
    try:
        import torch
        import torchani
        
        # Parse and prepare molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        
        # Generate initial 3D with RDKit
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
        
        # Get coordinates
        conf = mol.GetConformer()
        coords = []
        species = []
        
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
            species.append(atom.GetSymbol())
        
        # Check if all elements are supported by ANI-2x
        # ANI-2x supports: H, C, N, O, S, F, Cl
        supported = {'H', 'C', 'N', 'O', 'S', 'F', 'Cl'}
        if not all(s in supported for s in species):
            print(f"  Warning: Molecule contains unsupported elements for ANI-2x")
            # Fall back to MMFF optimization
            AllChem.MMFFOptimizeMolecule(mol)
            return mol
        
        # Load ANI-2x model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torchani.models.ANI2x(periodic_table_index=True).to(device)
        
        # Convert to tensors
        coords_tensor = torch.tensor([coords], dtype=torch.float32, device=device, requires_grad=True)
        species_tensor = torch.tensor([[model.species_to_tensor(s).item() for s in species]], device=device)
        
        # Simple gradient descent optimization
        optimizer = torch.optim.LBFGS([coords_tensor], lr=0.1, max_iter=100)
        
        def closure():
            optimizer.zero_grad()
            energy = model((species_tensor, coords_tensor)).energies
            energy.backward()
            return energy
        
        for _ in range(5):  # Multiple LBFGS iterations
            optimizer.step(closure)
        
        # Update molecule coordinates
        final_coords = coords_tensor.detach().cpu().numpy()[0]
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, final_coords[i].tolist())
        
        mol.SetProp('_Name', name)
        
        return mol
        
    except Exception as e:
        print(f"  ANI optimization failed: {e}")
        return None


def run_manual_ani_pipeline(
    molecules: List[Dict],
    output_dir: Path,
    use_gpu: bool = True
) -> List[Dict]:
    """
    Manual ANI-2x optimization pipeline when Auto3D is not available.
    """
    results = []
    
    try:
        import torch
        import torchani
        
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"Using device: {device}")
        
        # Load ANI-2x model once
        print("Loading ANI-2x neural network potential...")
        model = torchani.models.ANI2x(periodic_table_index=True).to(device)
        print("✓ Model loaded\n")
        
    except ImportError as e:
        print(f"ERROR: Required package not available: {e}")
        print("Please install: pip install torch torchani")
        return results
    
    sdf_dir = output_dir / 'sdf'
    pdb_dir = output_dir / 'pdb'
    sdf_dir.mkdir(exist_ok=True)
    pdb_dir.mkdir(exist_ok=True)
    
    for i, mol_data in enumerate(molecules):
        smiles = mol_data['SMILES']
        rank = int(mol_data['Dock_Rank'])
        name = f"DENOVO_EGFR_{rank}"
        
        print(f"[{i+1:3d}/{len(molecules)}] Rank {rank}: ", end='', flush=True)
        
        # Parse molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("FAILED (invalid SMILES)")
            continue
        
        mol = Chem.AddHs(mol)
        
        # Generate initial conformer with RDKit
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
        
        # Get species and check compatibility
        species = [atom.GetSymbol() for atom in mol.GetAtoms()]
        supported = {'H', 'C', 'N', 'O', 'S', 'F', 'Cl'}
        
        if not all(s in supported for s in species):
            unsupported = set(species) - supported
            print(f"SKIPPED (unsupported elements: {unsupported})")
            # Fall back to MMFF
            try:
                AllChem.MMFFOptimizeMolecule(mol)
                mol.SetProp('_Name', name)
                mol.SetProp('optimization', 'MMFF94s (fallback)')
                
                # Save
                sdf_path = sdf_dir / f'mol_{rank:03d}.sdf'
                writer = Chem.SDWriter(str(sdf_path))
                writer.write(mol)
                writer.close()
                
                results.append({
                    'Dock_Rank': rank,
                    'SMILES': smiles,
                    'Method': 'MMFF94s (fallback)',
                    'Status': 'Success'
                })
                print(f"✓ (MMFF fallback)")
            except:
                print("FAILED")
            continue
        
        try:
            # Get coordinates
            conf = mol.GetConformer()
            coords = [[conf.GetAtomPosition(i).x, 
                      conf.GetAtomPosition(i).y, 
                      conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())]
            
            # Convert to tensors
            coords_tensor = torch.tensor([coords], dtype=torch.float32, device=device, requires_grad=True)
            
            # Get species indices for ANI-2x
            species_map = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17}
            species_idx = [[species_map[s] for s in species]]
            species_tensor = torch.tensor(species_idx, device=device)
            
            # Calculate initial energy
            with torch.no_grad():
                initial_energy = model((species_tensor, coords_tensor)).energies.item()
            
            # Optimize with LBFGS
            optimizer = torch.optim.LBFGS([coords_tensor], lr=0.5, max_iter=20, 
                                          line_search_fn='strong_wolfe')
            
            def closure():
                optimizer.zero_grad()
                energy = model((species_tensor, coords_tensor)).energies
                energy.backward()
                return energy
            
            # Run optimization
            for step in range(10):
                loss = optimizer.step(closure)
            
            # Get final energy
            with torch.no_grad():
                final_energy = model((species_tensor, coords_tensor)).energies.item()
            
            # Update coordinates
            final_coords = coords_tensor.detach().cpu().numpy()[0]
            for j in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(j, final_coords[j].tolist())
            
            mol.SetProp('_Name', name)
            mol.SetProp('optimization', 'ANI-2x')
            mol.SetProp('initial_energy_hartree', f'{initial_energy:.6f}')
            mol.SetProp('final_energy_hartree', f'{final_energy:.6f}')
            
            # Convert to kcal/mol
            energy_kcal = final_energy * 627.509  # Hartree to kcal/mol
            
            # Save SDF
            sdf_path = sdf_dir / f'mol_{rank:03d}.sdf'
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol)
            writer.close()
            
            # Save PDB
            pdb_path = pdb_dir / f'mol_{rank:03d}.pdb'
            Chem.MolToPDBFile(mol, str(pdb_path))
            
            print(f"✓ {mol.GetNumAtoms()} atoms, E={final_energy:.4f} Ha")
            
            results.append({
                'Dock_Rank': rank,
                'SMILES': smiles,
                'Num_Atoms': mol.GetNumAtoms(),
                'ANI2x_Energy_Hartree': final_energy,
                'ANI2x_Energy_kcal': energy_kcal,
                'Method': 'ANI-2x',
                'Status': 'Success'
            })
            
        except Exception as e:
            print(f"FAILED ({e})")
            results.append({
                'Dock_Rank': rank,
                'SMILES': smiles,
                'Method': 'ANI-2x',
                'Status': f'Failed: {e}'
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Stage 2: High-Precision 3D with Auto3D/ANI-2x')
    parser.add_argument('--input', type=str, default='../top_docking_hits.csv',
                        help='Input CSV with SMILES')
    parser.add_argument('--output_dir', type=str, default='stage2_auto3d_ani2x',
                        help='Output directory')
    parser.add_argument('--n_mols', type=int, default=20,
                        help='Number of molecules to process')
    parser.add_argument('--engine', type=str, default='ANI2x',
                        choices=['ANI2x', 'ANI2xt', 'AIMNET'],
                        help='Neural network potential to use')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Disable GPU acceleration')
    args = parser.parse_args()
    
    print("=" * 60)
    print("STAGE 2: HIGH-PRECISION 3D STRUCTURE GENERATION")
    print("Method: Auto3D / ANI-2x Neural Network Potential")
    print("=" * 60)
    print()
    
    # Check installation
    print("Checking dependencies...")
    issues = check_auto3d_installation()
    
    if issues:
        print("\nDependency Status:")
        for issue in issues:
            print(f"  ⚠ {issue}")
        print()
    else:
        print("✓ All dependencies available\n")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load molecules
    print(f"Loading molecules from: {args.input}")
    molecules = []
    
    input_path = Path(args.input)
    if not input_path.exists():
        # Try relative to script location
        input_path = Path(__file__).parent / args.input
    
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            molecules.append(row)
    
    molecules = molecules[:args.n_mols]
    print(f"Processing {len(molecules)} molecules")
    print(f"Neural Network: {args.engine}")
    print(f"GPU: {'Disabled' if args.no_gpu else 'Enabled (if available)'}")
    print()
    
    print("-" * 60)
    
    # Run optimization
    if AUTO3D_AVAILABLE:
        print("Using Auto3D pipeline...\n")
        # Prepare input file
        smi_path = output_dir / 'input.smi'
        prepare_input_smi(molecules, str(smi_path))
        
        # Run Auto3D
        result_path = run_auto3d_optimization(
            str(smi_path),
            str(output_dir),
            k=1,
            optimizing_engine=args.engine,
            use_gpu=not args.no_gpu
        )
        
        if result_path:
            print(f"\n✓ Auto3D output: {result_path}")
    else:
        print("Auto3D not available, using manual ANI-2x pipeline...\n")
        results = run_manual_ani_pipeline(
            molecules,
            output_dir,
            use_gpu=not args.no_gpu
        )
        
        # Save summary
        if results:
            summary_path = output_dir / 'conversion_summary.csv'
            fieldnames = list(results[0].keys())
            
            with open(summary_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\n✓ Summary saved: {summary_path}")
            
            # Create combined SDF
            sdf_dir = output_dir / 'sdf'
            if sdf_dir.exists():
                combined_sdf = output_dir / 'all_molecules_ani2x.sdf'
                writer = Chem.SDWriter(str(combined_sdf))
                
                for sdf_file in sorted(sdf_dir.glob('*.sdf')):
                    suppl = Chem.SDMolSupplier(str(sdf_file))
                    for mol in suppl:
                        if mol is not None:
                            writer.write(mol)
                writer.close()
                print(f"✓ Combined SDF: {combined_sdf}")
            
            # Statistics
            successful = [r for r in results if r['Status'] == 'Success']
            print(f"\n✓ Successfully optimized: {len(successful)}/{len(molecules)}")
    
    print()
    print("=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)
    print()
    print("ANI-2x advantages over MMFF94s:")
    print("  • Quantum-accurate bond lengths and angles")
    print("  • Better treatment of non-bonded interactions")
    print("  • More reliable conformer energies")
    print("  • Improved docking pose predictions")


if __name__ == '__main__':
    main()
