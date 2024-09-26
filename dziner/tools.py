from langchain.agents import tool
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