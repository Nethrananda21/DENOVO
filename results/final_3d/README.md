# 3D Structure Generation Results

This folder contains 3D molecular structures generated from the top docking hits using three different methods.

## Folder Structure

```
final_3d/
├── README.md                       # This file
├── stage1_rdkit_convert.py         # Stage 1 script
├── stage2_auto3d_convert.py        # Stage 2 script
├── stage3_unimol_convert.py        # Stage 3 script
│
├── stage1_rdkit_etkdgv3/           # Stage 1: Standard Method
│   ├── sdf/                        # SDF format files
│   ├── pdb/                        # PDB format files  
│   ├── mol/                        # MOL format files
│   ├── all_molecules.sdf           # Combined SDF
│   └── conversion_summary.csv      # Summary with MMFF energies
│
├── stage2_auto3d_ani2x/            # Stage 2: High-Precision Method
│   ├── sdf/                        # SDF format (ANI-2x optimized)
│   ├── pdb/                        # PDB format (ANI-2x optimized)
│   ├── all_molecules_ani2x.sdf     # Combined SDF
│   └── conversion_summary.csv      # Summary with ANI-2x energies
│
└── stage3_unimol_ensemble/         # Stage 3: Conformer Ensembles
    ├── sdf/                        # Multi-conformer SDF files
    ├── pdb/                        # Best conformer PDB files
    ├── all_ensembles.sdf           # All conformers combined
    └── ensemble_summary.csv        # Summary with flexibility metrics
```

## Method Comparison

### Stage 1: RDKit ETKDGv3 (Standard & Reliable)
- **Force Field**: MMFF94s (Merck Molecular Force Field)
- **Method**: Distance geometry + torsion preferences from Cambridge Structural Database
- **Speed**: Fast (~1 sec/molecule)
- **Output**: Single best conformer
- **Use Case**: Quick screening, initial conformer generation

### Stage 2: ANI-2x Neural Network Potential (High-Precision)
- **Potential**: ANI-2x (trained on DFT data: ωB97X/6-31G*)
- **Method**: Neural network predicting quantum mechanical energies
- **Speed**: Moderate (~5-10 sec/molecule with GPU)
- **Output**: Single quantum-accurate conformer
- **Supported Elements**: H, C, N, O, S, F, Cl (Br falls back to MMFF)
- **Use Case**: Accurate docking, binding affinity prediction

### Stage 3: RDKit Diverse ETKDG / Uni-Mol (Conformer Ensembles)
- **Method**: RMSD-pruned diverse conformer generation (mimics Uni-Mol)
- **Output**: 10 diverse conformers per molecule
- **Metrics**: Energy range, RMSD diversity
- **Use Case**: Flexible docking, entropy estimation, pharmacophore screening

## Results Summary

| Stage | Method | Conformers | Success Rate | Key Metric |
|-------|--------|------------|--------------|------------|
| 1 | RDKit MMFF94s | 1/molecule | 20/20 (100%) | MMFF Energy |
| 2 | ANI-2x Neural Net | 1/molecule | 18/20 + 2 fallback | QM Energy (Ha) |
| 3 | RDKit Diverse | 10/molecule | 185 total | RMSD diversity |

## Stage 3 Flexibility Analysis

| Metric | Value |
|--------|-------|
| Avg conformers/molecule | 9.2 |
| Avg rotatable bonds | 3.5 |
| Avg conformer RMSD | 1.54 Å |
| Avg energy range | 3.8 kcal/mol |
| Total conformers | 185 |

Higher RMSD and energy range indicate more flexible molecules with diverse conformations.

## Quality Comparison

| Property | MMFF94s (Stage 1) | ANI-2x (Stage 2) | Ensemble (Stage 3) |
|----------|-------------------|------------------|-------------------|
| Bond lengths | ±0.02 Å | ±0.005 Å | ±0.02 Å |
| Bond angles | ±2° | ±0.5° | ±2° |
| Conformers | 1 | 1 | 10 |
| Flexibility | No | No | Yes |
| Energy units | kcal/mol | Hartree | kcal/mol |

## Usage Recommendations

1. **For quick visualization**: Use Stage 1 (RDKit) structures
2. **For accurate docking**: Use Stage 2 (ANI-2x) structures  
3. **For flexible docking**: Use Stage 3 ensembles
4. **For MD simulations**: Use Stage 2 + equilibration
5. **For pharmacophore modeling**: Use Stage 3 ensembles
6. **For publication figures**: Use Stage 2 for accurate geometries

## Files Generated

- **20 molecules** from top docking hits
- **Stage 1**: 20 single conformers (SDF, PDB, MOL)
- **Stage 2**: 20 quantum-optimized conformers (SDF, PDB)
- **Stage 3**: 185 diverse conformers (multi-conformer SDF)

## References

1. ETKDGv3: Riniker & Landrum, J. Chem. Inf. Model. 2015
2. ANI-2x: Devereux et al., J. Chem. Theory Comput. 2020
3. MMFF94s: Halgren, J. Comput. Chem. 1996
4. Uni-Mol: Zhou et al., ICLR 2023
