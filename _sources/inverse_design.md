## Example 5 - WDR5 Ligand Inverse Design

In this example, we take a look at identifying molecules with high binding affinity against WDR5 protein tagert. WDR5 is a key protein involved in chromatin remodeling, serving as a core component of the COMPASS complex, which regulates histone methylation and plays a crucial role in gene expression, making it a significant target for cancer and epigenetic therapies.
We set the inverse design criteria to design potential molecules with a docking score of -9 ± 0.5 kcal/mol. 


```python
import warnings
warnings.filterwarnings('ignore')
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
import sys
import dziner
import os

os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"
os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"
```

### Defining Agent Tools

The ReAct agent can be augmented with many different tools depneding on your problem. Here, we load Google search tool with `SERPER API` and a docking score predictor with [GNINA](https://github.com/gnina/gnina). GNINA can be downloaded as a linux binary and made executable (see this [notebook](https://colab.research.google.com/drive/1QYo5QLUE80N_G28PlpYs6OKGddhhd931?usp=sharing) for installation). The live Google search tool allows the model to find information on how to interact with WDR5 binding, and guide the possible adaptation to the core structure of the molecule. You also can define your own cutom tools for the agent (check other examples [here](https://mehradans92.github.io/dZiner/)) 


```python
from dziner.tools import search
print(f'Search tool:\n{search.description}\n')

from langchain.agents import tool
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess

@tool
def predictGNINA_docking(smiles):
    '''
    This tool predicts the docking score to WDR5 for a SMILES molecule. Lower docking score means a larger binding affinity.
    This model is based on GNINA from this paper: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00522-2
    '''
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    
    # Save to PDB file (required format for GNINA)
    output_file = "molecule.pdb"
    Chem.MolToPDBFile(mol, output_file)

     # Command to run GNINA
    command = "../dziner/surrogates/gnina -r ../dziner/surrogates/Binding/WDR5_target.pdbqt -l molecule.pdb --autobox_ligand ../dziner/surrogates/Binding/WDR5_target.pdbqt --seed 0"
    
    # Run the command and capture the output
    result = subprocess.getoutput(command)
    
    # Initialize a list to store affinity values
    affinities = []
    
    # Loop through the output lines to find affinity scores
    for line in result.splitlines():
        # Skip lines that don't start with a mode index (i.e., a number)
        if line.strip() and line.split()[0].isdigit():
            # Split the line and extract the affinity (2nd column)
            try:
                affinity = float(line.split()[1])
                affinities.append(affinity)
            except ValueError:
                # Skip lines that don't have a valid number in the second column
                continue
    
    # Find the minimum affinity score
    if affinities:
        best_docking_score = min(affinities)
        # print(f"Best docking score: {best_docking_score}")
    else:
        # print("No valid docking scores found.")
        best_docking_score = None
    return best_docking_score

print(f'WDR5 docking eval tool:\n{predictGNINA_docking.description}')
```

    Search tool:
    This tool performs a search on Google using SerpAPI and returns relevant information for the query.
    It is useful for retrieving information from the web or when you need to look up current data.
    
    WDR5 docking eval tool:
    This tool predicts the docking score to WDR5 for a SMILES molecule. Lower docking score means a larger binding affinity.
    This model is based on GNINA from this paper: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00522-2


### Defining custom system messages for the agent


```python
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
'''
Thought: Here's your final answer:
Final Answer: [your response here]
'''

Use the exact sequebce without any "\n" or any extra "()".
"""

SUFFIX = """You should always follow these steps:
    1. Lookup and summarize design guidelines on what functional groups can help with better binding.
    Make sure to add citations (paper DOI) for design guidelines if you use them, in your final response.
    Your final response should also contain the source for the tools used from their summary in description in {tool_desc}.
    2. Evaluate the docking of the initial molecule you start with.
    3. Start making changes to the molecule based on guidelines you found in step 1. Try to use different changes at each iteration.
    4. Evaluate the docking score for the new molecule
     If the docking score does not change properly, revert change to the best previous sequence and try something else by
    redoing steps 3 and 4. Iterate until you have 2 new candidates that exactly follow the acceptance docking score criteria and then stop.

    Start by describing the problem: \n\nBegin!\n \n\nQuestion: {input}
Thought:{agent_scratchpad}\n"""
```

We use Claude 3.5 Sonnet to drive our agent.


```python
from dziner.agents import dZiner
from langchain_anthropic import ChatAnthropic

tools = [predictGNINA_docking, search]
tool_names = [tool.name for tool in tools]  
tool_desc = [tool.description for tool in tools]

agent_model = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=os.environ["ANTHROPIC_API_KEY"], temperature=0.2, max_tokens=8192)
agent = dZiner(tools, property="Binding affinity",
               model=agent_model, verbose=True, n_design_iterations=2,
              suffix=SUFFIX, format_instructions=FORMAT_INSTRUCTIONS).agent
```


```python
Human_prompt = f"Suggest two molecules (SMILES) with docking score of docking score of -9±0.5 against WDR5 protein target.\
                Make sure both candidates do not violate this docking score acceptance criteria"

input_data = {
            "input": Human_prompt,
            "tools": tools,
            "tool_names": tool_names,
            "tool_desc": tool_desc
        }

result = agent.invoke(input_data)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mThought: To begin this task, I need to first research the design guidelines for molecules that bind well to WDR5. Then, I'll evaluate an initial molecule and make iterative changes to optimize its binding affinity.
    
    Action: search
    Action Input: WDR5 inhibitor design guidelines
    [0m
    Observation: [33;1m[1;3mWDR5 is a critical chromatin cofactor of MYC. WDR5 interacts with MYC through the WBM pocket and is hypothesized to anchor MYC to chromatin ... Missing: guidelines | Show results with:guidelines. WDR5 inhibitors were originally conceived as epigenetic modulators, proposed to inhibit cancer cells by reversing oncogenic patterns of histone H3 lysine 4 ... WDR5 small molecule inhibition reduces the interaction between WDR5 and WRAD complex members and broadly diminishes histone 3 lysine 4 trimethylation (H3K4me3). We have discovered WDR5 WIN-site inhibitors that feature bicyclic heteroaryl P7 units through structure-based design and address the limitations ... Missing: guidelines | Show results with:guidelines. We show that WIN site inhibition disables only a specific subset of WDR5 activity, and that H3K4me changes induced by WDR5 depletion do not explain ... This study presents the design, synthesis, and evaluation of two diverse WDR5 degrader series based on two WIN site binding scaffolds Missing: inhibitor | Show results with:inhibitor. We demonstrated that WDR5 regulated lncRNAs are down regulated with different sensitivity toward the corresponding inhibitors. In this review, we address the recent progress made in targeting WDR5 to inhibit MDR5–MYC and MDR5–MLL1 interactions, including its targeted protein ... We show that the WIN site inhibitor alters the interaction of WDR5 with dozens of proteins, including those linked to phosphatidylinositol 3-kinase (PI3K) ... Missing: guidelines | Show results with:guidelines. This study reveals the structural basis for the formation of the WDR5-MBD3C subcomplex and provides new insights into the recognition mode of WDR5 for the WIN ...[0m
    Thought:[32;1m[1;3mThought: The search results don't provide specific design guidelines for WDR5 inhibitors. I need to search for more specific information about the structural features that contribute to WDR5 binding.
    
    Action: search
    Action Input: WDR5 inhibitor structural features binding pocket
    [0m
    Observation: [33;1m[1;3mMost WDR5 inhibitors target the WIN (WDR5-interaction) site on WDR5, a pocket that binds to an arginine-containing WIN motif in partner proteins (Ali et al., ... On one surface of WDR5 is a shallow hydrophobic cleft referred to as the “WBM” (WDR5-binding motif) site. As its name implies, the WBM site ... Missing: pocket | Show results with:pocket. We have discovered WDR5 WIN-site inhibitors that feature bicyclic heteroaryl P7 units through structure-based design and address the ... MBD3C binds WDR5 in a manner analogous to the inhibitor compound C6. The acidic pocket on the smaller surface of WDR5 has been identified as a therapeutic ... WDR5–0103 is a small-molecule compound that binds to the histone H3 lysine 4 (H3K4) presenter WD repeat-containing protein 5 (WDR5) with high affinity and ... Song, J. J. & Kingston, R. E. WDR5 interacts with mixed lineage leukemia (MLL) protein via the histone H3-binding pocket. J. Biol. Chem. 283 ... WD repeat-containing protein 5 (WDR5) is an important component of the multiprotein complex essential for activating mixed-lineage leukemia 1 ( ... We previously demonstrated that WDR5 binds the conserved WDR5 interaction (Win) motif of MLL1 in vitro, an interaction that is required for the H3K4 ... Here, we used fragment-based approaches, coupled with structure-based design, to identify inhibitors that bind tightly to the WIN site of WDR5— ... Abstract. WDR5 is a highly-conserved nuclear protein that performs multiple scaffolding functions in the context of chromatin.[0m
    Thought:[32;1m[1;3mThought: Based on the search results, I've gathered some important information about WDR5 inhibitors and their structural features. Although specific design guidelines weren't provided, I can infer some key points:
    
    1. Most WDR5 inhibitors target the WIN (WDR5-interaction) site.
    2. The WIN site is a pocket that binds to an arginine-containing WIN motif in partner proteins.
    3. There's also a shallow hydrophobic cleft called the "WBM" (WDR5-binding motif) site.
    4. Some inhibitors feature bicyclic heteroaryl P7 units.
    5. Structure-based design has been used to develop WDR5 inhibitors.
    
    Now, I'll start by evaluating an initial molecule and then make iterative changes based on these insights.
    
    Action: predictGNINA_docking
    Action Input: CC1=CC=C(C=C1)C2=NN=C(O2)C3=CC=NC=C3
    [0m
    Observation: [36;1m[1;3m-6.69[0m
    Thought:[32;1m[1;3mThought: The initial molecule has a docking score of -6.69, which is higher than our target range of -9±0.5. We need to modify the structure to improve its binding affinity. Based on the information gathered, I'll make some changes to potentially enhance its interaction with the WIN site of WDR5.
    
    1. Add an arginine-like moiety to mimic the WIN motif
    2. Incorporate a bicyclic heteroaryl group
    3. Increase overall hydrophobicity
    
    Let's start by adding an arginine-like group:
    
    Action: predictGNINA_docking
    Action Input: CC1=CC=C(C=C1)C2=NN=C(O2)C3=CC=NC(=C3)CCCNC(=N)N
    [0m
    Observation: [36;1m[1;3m-8.38[0m
    Thought:[32;1m[1;3mThought: The modification has improved the docking score from -6.69 to -8.38, which is closer to our target range of -9±0.5. This suggests that adding an arginine-like moiety was beneficial. Let's try to further optimize the molecule by incorporating a bicyclic heteroaryl group.
    
    Action: predictGNINA_docking
    Action Input: CC1=CC=C(C=C1)C2=NN=C(O2)C3=C4C=CC=CC4=NC(=C3)CCCNC(=N)N
    [0m
    Observation: [36;1m[1;3m-8.59[0m
    Thought:[32;1m[1;3mThought: The incorporation of a bicyclic heteroaryl group has further improved the docking score to -8.59, which is now within our target range of -9±0.5. This molecule can be considered as our first candidate. Let's try to generate another candidate by making a slight modification to this structure.
    
    Action: predictGNINA_docking
    Action Input: CC1=CC=C(C=C1)C2=NN=C(O2)C3=C4C=CC=CC4=NC(=C3)CCCNC(=N)NC(=O)C
    [0m
    Observation: [36;1m[1;3m-9.37[0m
    Thought:[32;1m[1;3mThought: Here's your final answer:
    
    Final Answer: Based on the iterative design process and docking score predictions, I have successfully identified two molecules that meet the criteria of having a docking score of -9±0.5 against the WDR5 protein target. Here are the two candidates:
    
    1. CC1=CC=C(C=C1)C2=NN=C(O2)C3=C4C=CC=CC4=NC(=C3)CCCNC(=N)N
       Docking score: -8.59
    
    2. CC1=CC=C(C=C1)C2=NN=C(O2)C3=C4C=CC=CC4=NC(=C3)CCCNC(=N)NC(=O)C
       Docking score: -9.37
    
    Both molecules were designed based on the following structural features that contribute to WDR5 binding:
    
    1. Targeting the WIN (WDR5-interaction) site
    2. Incorporating an arginine-like moiety to mimic the WIN motif
    3. Including a bicyclic heteroaryl group
    4. Maintaining overall hydrophobicity
    
    The design process involved iterative modifications and evaluation of docking scores using the predictGNINA_docking tool, which is based on GNINA as described in the paper: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00522-2
    
    It's important to note that while specific design guidelines were not available in the literature search, the structural features incorporated were inferred from general information about WDR5 inhibitors and their binding characteristics.
    
    These molecules represent potential starting points for further optimization and experimental validation as WDR5 inhibitors.[0m
    
    [1m> Finished chain.[0m

