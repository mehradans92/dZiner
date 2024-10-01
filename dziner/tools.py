from langchain.agents import tool
import dziner.sascorer
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
import os


from dockstring import load_target

@tool
def predict_docking_score_DRD2(smiles):
    '''
    This tool predicts the docking score to DRD2 for a SMILES molecule. Lower docking score means a larger binding affinity.
    This model is based on Autodock Vina from this paper: https://pubs.acs.org/doi/10.1021/acs.jcim.1c01334 
    '''
    ## The predict_docking_score_DRD2 is based on this paper: https://pubs.acs.org/doi/10.1021/acs.jcim.1c01334
    ## And is coming from this URL 
    ## https://github.com/dockstring/dockstring/tree/main
    ## Also see https://github.com/dockstring/dockstring/blob/main/dockstring/target.py#L160

    target = load_target("DRD2")
    try:
        score, affinities = target.dock(smiles)
    except:
        score = 'Invalid molecule'
    return score

from langchain_community.utilities import GoogleSerperAPIWrapper

@tool
def search(query):
    '''
    This tool performs a search on Google using SerpAPI and returns relevant information for the query.
    It is useful for retrieving information from the web or when you need to look up current data.
    '''
    search = GoogleSerperAPIWrapper(engine='google', serpapi_api_key=os.getenv("SERPER_API_KEY"))
    results = search.run(query)
    return results

@tool
def mol_chemical_feasibility(smiles: str):
    '''
    This tool inputs a SMILES of a molecule and outputs chemical feasibility, Synthetic Accessibility (SA) scores,
      molecular weight, and the SMILES. SA the ease of synthesis of compounds according to their synthetic complexity
        which combines starting materials information and structural complexity. Lower SA, means better synthesiability.
    '''
    smiles = smiles.replace("\n", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES", 0, 0, smiles

    # Calculate SA score
    sa_score = dziner.sascorer.calculateScore(mol)

    # Optionally calculate QED score
    molecular_weight = Descriptors.MolWt(mol)
    return "Valid SMILES", sa_score, molecular_weight, smiles

@tool
def drug_chemical_feasibility(smiles: str):
    '''
    This tool inputs a SMILES of a drug candidate and outputs chemical feasibility, Synthetic Accessibility (SA),
      Quantentive drug-likeliness(QED) scores, molecular weight, and the SMILES. SA the ease of synthesis of compounds
        according to their synthetic complexity which combines starting materials information and structural complexity. 
        Lower SA, means better synthesiability. QED combines eight physicochemical properties(molecular weight, LogP, 
        H-bond donors, H-bond acceptors, charge, aromaticity, stereochemistry and solubility), generating a score between 0 and 1.
    '''
    smiles = smiles.replace("\n", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES", 0, 0, smiles

    # Calculate SA score
    sa_score = dziner.sascorer.calculateScore(mol)

    # Optionally calculate QED score
    molecular_weight = Descriptors.MolWt(mol)
    qed_score = QED.qed(mol)
    return "Valid SMILES", sa_score, molecular_weight, qed_score, smiles