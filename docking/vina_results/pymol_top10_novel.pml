# PyMOL Visualization Script - Top 10 NOVEL EGFR Inhibitor Candidates
# =====================================================================
# Ranked by AutoDock Vina affinity (excluding known compounds)
# Target: EGFR kinase (PDB: 1M17)
# Reference: Erlotinib (native ligand)

# Reinitialize PyMOL
reinitialize

# =============================================================================
# LOAD RECEPTOR AND REFERENCE
# =============================================================================

# Load EGFR receptor
load C:/DENOVO/docking/1M17.pdb, EGFR_receptor

# Remove waters
remove resn HOH

# Color receptor
color gray80, EGFR_receptor
hide everything, EGFR_receptor
show cartoon, EGFR_receptor
set cartoon_transparency, 0.7, EGFR_receptor

# =============================================================================
# LOAD TOP 10 NOVEL CANDIDATES (Vina-ranked, known compounds excluded)
# =============================================================================

# Rank 1: mol_003 - BEST NOVEL (-9.20 kcal/mol)
load C:/DENOVO/docking/vina_results/poses_pdb/mol_003_pose1.pdb, mol_003_BEST
color green, mol_003_BEST
show sticks, mol_003_BEST

# Rank 2: mol_016 (-9.02 kcal/mol)
load C:/DENOVO/docking/vina_results_extended/poses_pdb/mol_016_pose1.pdb, mol_016
color cyan, mol_016
show sticks, mol_016

# Rank 3: mol_012 (-8.90 kcal/mol)
load C:/DENOVO/docking/vina_results_extended/poses_pdb/mol_012_pose1.pdb, mol_012
color yellow, mol_012
show sticks, mol_012

# Rank 4: mol_001 (-8.78 kcal/mol) - Quinazoline scaffold
load C:/DENOVO/docking/vina_results/poses_pdb/mol_001_pose1.pdb, mol_001
color orange, mol_001
show sticks, mol_001

# Rank 5: mol_008 (-8.53 kcal/mol) - Highest QED
load C:/DENOVO/docking/vina_results_extended/poses_pdb/mol_008_pose1.pdb, mol_008
color magenta, mol_008
show sticks, mol_008

# Rank 6: mol_019 (-8.46 kcal/mol)
load C:/DENOVO/docking/vina_results_extended/poses_pdb/mol_019_pose1.pdb, mol_019
color salmon, mol_019
show sticks, mol_019

# Rank 7: mol_011 (-8.39 kcal/mol)
load C:/DENOVO/docking/vina_results_extended/poses_pdb/mol_011_pose1.pdb, mol_011
color palegreen, mol_011
show sticks, mol_011

# Rank 8: mol_006 (-8.37 kcal/mol)
load C:/DENOVO/docking/vina_results_extended/poses_pdb/mol_006_pose1.pdb, mol_006
color lightblue, mol_006
show sticks, mol_006

# Rank 9: mol_013 (-8.32 kcal/mol)
load C:/DENOVO/docking/vina_results_extended/poses_pdb/mol_013_pose1.pdb, mol_013
color wheat, mol_013
show sticks, mol_013

# Rank 10: mol_020 (-8.23 kcal/mol)
load C:/DENOVO/docking/vina_results_extended/poses_pdb/mol_020_pose1.pdb, mol_020
color lightorange, mol_020
show sticks, mol_020

# =============================================================================
# BINDING SITE RESIDUES
# =============================================================================

# Select key binding site residues
select binding_site, EGFR_receptor and resi 766+769+770+771+772+773+790+791+792+793+794+795+797+817+818+835+836+838+853+854+855

# Show binding site as sticks
show sticks, binding_site
color lightpink, binding_site and elem C
util.cnc binding_site

# Label key residues
label binding_site and name CA and resi 790+793+797+817+835+838, "%s%s" % (resn, resi)
set label_color, black
set label_size, 14

# =============================================================================
# H-BOND DETECTION FOR ALL LIGANDS
# =============================================================================

# H-bonds for mol_003 (BEST)
dist hbond_003, mol_003_BEST, binding_site, mode=2, cutoff=3.5
color green, hbond_003

# H-bonds for mol_016
dist hbond_016, mol_016, binding_site, mode=2, cutoff=3.5
color cyan, hbond_016

# H-bonds for mol_012
dist hbond_012, mol_012, binding_site, mode=2, cutoff=3.5
color yellow, hbond_012

# H-bonds for mol_001 (quinazoline scaffold with carboxylic acid)
# Verified H-bond: 3.0 Å to THR830-OG1 (threonine hydroxyl)

# Select H-bond partner residue
select THR830, EGFR_receptor and resi 830

# Show H-bond partner
show sticks, THR830
color tv_orange, THR830 and elem C

# Measure H-bond using mode=2 for proper detection
dist hbond_001_THR830, mol_001, THR830, mode=2, cutoff=3.5

color orange, hbond_001_THR830
set dash_width, 3, hbond_001_THR830

# Label H-bond partner
label THR830 and name OG1, "THR830 (3.0A)"

# H-bonds for mol_008
dist hbond_008, mol_008, binding_site, mode=2, cutoff=3.5
color magenta, hbond_008

# H-bonds for mol_019
dist hbond_019, mol_019, binding_site, mode=2, cutoff=3.5
color salmon, hbond_019

# H-bonds for mol_011
dist hbond_011, mol_011, binding_site, mode=2, cutoff=3.5
color palegreen, hbond_011

# H-bonds for mol_006
# Verified H-bond: 3.5 Å to THR766-OG1 (threonine hydroxyl)
select THR766, EGFR_receptor and resi 766
show sticks, THR766
color tv_blue, THR766 and elem C
dist hbond_006_THR766, mol_006, THR766, mode=2, cutoff=4.0
color lightblue, hbond_006_THR766
set dash_width, 3, hbond_006_THR766
label THR766 and name OG1, "THR766 (3.5A)"

# H-bonds for mol_013
dist hbond_013, mol_013, binding_site, mode=2, cutoff=3.5
color wheat, hbond_013

# H-bonds for mol_020
dist hbond_020, mol_020, binding_site, mode=2, cutoff=3.5
color lightorange, hbond_020

# =============================================================================
# VIEW SETTINGS
# =============================================================================

# Center on binding site
center binding_site
zoom binding_site, 8

# Disable clipping
set depth_cue, 0
set fog, 0
clip slab, 50

# Background
bg_color white

# Ray tracing settings
set ray_shadows, 1
set antialias, 2
set ray_trace_mode, 1

# =============================================================================
# GROUP LIGANDS FOR EASY TOGGLING
# =============================================================================

group top3_leads, mol_003_BEST mol_016 mol_012
group quinazoline, mol_001
group novel_candidates, mol_008 mol_019 mol_011 mol_006 mol_013 mol_020
group all_hbonds, hbond_*

# =============================================================================
# DISPLAY OPTIONS
# =============================================================================

# Initially show only top 3
disable novel_candidates
enable top3_leads
enable quinazoline

# =============================================================================
# PREDEFINED VIEWS
# =============================================================================

# Save current view
set_view (\
     0.980226040,    0.113399386,   -0.162182897,\
    -0.060825769,    0.944879711,    0.321765423,\
     0.189751863,   -0.305610687,    0.932927132,\
     0.000000000,    0.000000000,  -65.000000000,\
    22.000000000,    0.300000000,   52.800003052,\
    50.000000000,   80.000000000,  -20.000000000 )

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

# Print instructions
print ""
print "============================================================"
print "TOP 10 NOVEL EGFR INHIBITOR CANDIDATES"
print "============================================================"
print ""
print "Color Legend:"
print "  GREEN     = mol_003 (BEST, -9.20 kcal/mol)"
print "  CYAN      = mol_016 (-9.02 kcal/mol)"
print "  YELLOW    = mol_012 (-8.90 kcal/mol)"
print "  ORANGE    = mol_001 (-8.78 kcal/mol, H-bond: THR830 3.0A)"
print "  MAGENTA   = mol_008 (-8.53 kcal/mol, highest QED)"
print "  SALMON    = mol_019 (-8.46 kcal/mol)"
print "  PALEGREEN = mol_011 (-8.39 kcal/mol)"
print "  LIGHTBLUE = mol_006 (-8.37 kcal/mol, H-bond: THR766 3.5A)"
print "  WHEAT     = mol_013 (-8.32 kcal/mol)"
print "  LIGHTORANGE = mol_020 (-8.23 kcal/mol)"
print ""
print "Groups available:"
print "  top3_leads       - Best 3 candidates"
print "  quinazoline      - mol_001 (EGFR-like scaffold)"
print "  novel_candidates - Remaining candidates"
print "  all_hbonds       - All H-bond measurements"
print ""
print "Commands:"
print "  enable/disable <group>  - Toggle visibility"
print "  show sticks, <mol>      - Show molecule"
print "  hide sticks, <mol>      - Hide molecule"
print "  ray                     - Render high-quality image"
print "============================================================"
