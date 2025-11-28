#!/usr/bin/env python3
"""
PubChem Novelty Verification for Top 10 Candidates
===================================================
Searches PubChem via REST API using InChI Keys
"""

import urllib.request
import urllib.error
import json
import time
import csv
from pathlib import Path

# =============================================================================
# TOP 10 CANDIDATES (from comprehensive analysis)
# =============================================================================

TOP_10 = [
    {"rank": 1, "mol_id": "mol_003", "vina": -9.20, "inchi_key": "ZDRZRGXDXFPNEE-UHFFFAOYSA-N", "smiles": "COc1ccc(O)c(C(=O)Nc2ccc(C#N)c(C(F)(F)F)c2)c1"},
    {"rank": 2, "mol_id": "mol_016", "vina": -9.02, "inchi_key": "KDIKMAZUAYHUBQ-UHFFFAOYSA-N", "smiles": "COc1ccc(C(=O)Nc2ccccc2C(C)(C)C)c(O)c1C"},
    {"rank": 3, "mol_id": "mol_012", "vina": -8.90, "inchi_key": "DPDYLFCYFPNHAZ-UHFFFAOYSA-N", "smiles": "COc1ccc(C(=O)Nc2ncccc2Cl)c(OC)c1OC"},
    {"rank": 4, "mol_id": "mol_018", "vina": -8.89, "inchi_key": "JVGYUSDCOGHOCH-UHFFFAOYSA-N", "smiles": "CC1CCN(CC(=O)N2c3ccccc3NC(=O)c3ccccc32)CC1"},
    {"rank": 5, "mol_id": "mol_001", "vina": -8.78, "inchi_key": "JONWPAQBNZBUFV-UHFFFAOYSA-N", "smiles": "COc1cc2ncnc(NC3CCC(C(=O)O)CC3)c2cc1OC"},
    {"rank": 6, "mol_id": "mol_008", "vina": -8.53, "inchi_key": "WZSGMVBRGXDRCE-UHFFFAOYSA-N", "smiles": "Cc1ccc(-c2cc(C3=NN(C(N)=O)CC3)sc2C(C)C)cn1"},
    {"rank": 7, "mol_id": "mol_002", "vina": -8.52, "inchi_key": "AXJMZPARUVLPCE-UHFFFAOYSA-N", "smiles": "COc1cc2ncnc(NC3CCCCC3)c2cc1OC"},
    {"rank": 8, "mol_id": "mol_015", "vina": -8.51, "inchi_key": "XKYJZVUQHVNEJF-UHFFFAOYSA-N", "smiles": "COc1ccc(-c2ccc(C(N)=O)cc2)cc1OC1CCCC1"},
    {"rank": 9, "mol_id": "mol_019", "vina": -8.46, "inchi_key": "OTONUPHHEONBOR-UHFFFAOYSA-N", "smiles": "COc1ccc2c(c1)N(CCN1CCOc3c(F)cccc3C1)C2=O"},
    {"rank": 10, "mol_id": "mol_011", "vina": -8.39, "inchi_key": "OJIKBCRXWLUFMV-UHFFFAOYSA-N", "smiles": "CC(C)C1CCC(N2CCC(Nc3nc4ccccc4s3)CC2)C1=O"},
]

OUTPUT_DIR = Path(r"C:\DENOVO\results")

# =============================================================================
# PUBCHEM SEARCH FUNCTIONS
# =============================================================================

def search_pubchem_by_inchikey(inchi_key):
    """Search PubChem for compound using InChI Key."""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey"
    
    try:
        # Search for compound
        url = f"{base_url}/{inchi_key}/cids/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=30)
        data = json.loads(response.read().decode())
        
        if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
            cids = data['IdentifierList']['CID']
            if cids:
                return {'found': True, 'cid': cids[0], 'all_cids': cids}
        
        return {'found': False, 'cid': None, 'all_cids': []}
        
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {'found': False, 'cid': None, 'all_cids': [], 'status': 'NOT_FOUND'}
        return {'found': False, 'cid': None, 'all_cids': [], 'error': str(e)}
    except Exception as e:
        return {'found': False, 'cid': None, 'all_cids': [], 'error': str(e)}


def get_pubchem_compound_info(cid):
    """Get detailed compound information from PubChem."""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
    
    try:
        # Get basic properties
        url = f"{base_url}/{cid}/property/MolecularFormula,MolecularWeight,IUPACName,Title/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=30)
        data = json.loads(response.read().decode())
        
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
            props = data['PropertyTable']['Properties'][0]
            return {
                'formula': props.get('MolecularFormula', ''),
                'mw': props.get('MolecularWeight', ''),
                'iupac': props.get('IUPACName', ''),
                'title': props.get('Title', '')
            }
        
        return {}
        
    except Exception as e:
        return {'error': str(e)}


# =============================================================================
# MAIN
# =============================================================================

print("=" * 80)
print("PUBCHEM NOVELTY VERIFICATION - TOP 10 CANDIDATES")
print("=" * 80)
print("\nSearching PubChem database for each compound...")
print("(If NOT FOUND → compound is NOVEL)\n")

results = []

for candidate in TOP_10:
    rank = candidate['rank']
    mol_id = candidate['mol_id']
    vina = candidate['vina']
    inchi_key = candidate['inchi_key']
    smiles = candidate['smiles']
    
    print(f"[{rank:2d}] {mol_id} (Vina: {vina:.2f} kcal/mol)")
    print(f"    InChI Key: {inchi_key}")
    print(f"    Searching PubChem...", end=' ')
    
    # Search PubChem
    search_result = search_pubchem_by_inchikey(inchi_key)
    time.sleep(0.5)  # Rate limiting
    
    if search_result['found']:
        cid = search_result['cid']
        print(f"FOUND! CID: {cid}")
        
        # Get additional info
        info = get_pubchem_compound_info(cid)
        time.sleep(0.3)
        
        if info and 'title' in info:
            print(f"    Name: {info.get('title', 'Unknown')}")
        
        status = "KNOWN"
        pubchem_cid = cid
        pubchem_name = info.get('title', '') if info else ''
    else:
        print("NOT FOUND → ✓ NOVEL COMPOUND!")
        status = "NOVEL"
        pubchem_cid = None
        pubchem_name = ""
    
    results.append({
        'Vina_Rank': rank,
        'Mol_ID': mol_id,
        'Vina_Affinity': vina,
        'InChI_Key': inchi_key,
        'SMILES': smiles,
        'PubChem_Status': status,
        'PubChem_CID': pubchem_cid,
        'PubChem_Name': pubchem_name
    })
    
    print()

# =============================================================================
# SAVE RESULTS
# =============================================================================

output_file = OUTPUT_DIR / 'pubchem_novelty_verification.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['Vina_Rank', 'Mol_ID', 'Vina_Affinity', 'InChI_Key', 'SMILES', 
                  'PubChem_Status', 'PubChem_CID', 'PubChem_Name']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"✓ Results saved to: {output_file}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("NOVELTY SUMMARY")
print("=" * 80)

novel = [r for r in results if r['PubChem_Status'] == 'NOVEL']
known = [r for r in results if r['PubChem_Status'] == 'KNOWN']

print(f"\n  NOVEL compounds: {len(novel)}")
print(f"  KNOWN compounds: {len(known)}")

if novel:
    print("\n  ✓ NOVEL CANDIDATES (Recommended for Further Development):")
    for r in novel:
        print(f"    - {r['Mol_ID']}: Vina {r['Vina_Affinity']:.2f} kcal/mol")

if known:
    print("\n  ✗ KNOWN COMPOUNDS (Already in databases):")
    for r in known:
        name = r['PubChem_Name'][:30] + "..." if len(r.get('PubChem_Name', '')) > 30 else r.get('PubChem_Name', 'Unknown')
        print(f"    - {r['Mol_ID']}: CID {r['PubChem_CID']} ({name})")

# =============================================================================
# FINAL RECOMMENDATIONS
# =============================================================================

print("\n" + "=" * 80)
print("FINAL RECOMMENDATIONS")
print("=" * 80)

# Filter novel candidates with good affinity
novel_good = [r for r in results if r['PubChem_Status'] == 'NOVEL' and r['Vina_Affinity'] < -8.5]

if novel_good:
    print("\n🏆 TOP NOVEL LEAD CANDIDATES (Good affinity + Novel structure):\n")
    for r in novel_good:
        print(f"  {r['Mol_ID']}:")
        print(f"    Vina Affinity: {r['Vina_Affinity']:.2f} kcal/mol")
        print(f"    InChI Key: {r['InChI_Key']}")
        print(f"    SMILES: {r['SMILES']}")
        print()
else:
    print("\n  No candidates found with both good affinity (<-8.5) and novel structure.")
    print("  Consider expanding search criteria or generating more candidates.")

print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
