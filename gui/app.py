import streamlit as st
st.set_page_config(layout="wide")

import warnings
from langchain._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings('ignore', category=LangChainDeprecationWarning)

api_key_url = (
    "https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key"
)

from langchain.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import dziner
import tempfile
import os
import dziner.sascorer
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
from streamlit.components.v1 import html
import base64

import io
import sys
from contextlib import redirect_stdout, redirect_stderr


from langchain.agents import tool
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
from langchain_community.tools import HumanInputRun
from langchain_anthropic import ChatAnthropic
from dziner.agents import dZiner
import dziner

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
        if line.strip() and line.split()[0].isdigit():
            try:
                affinity = float(line.split()[1])
                affinities.append(affinity)
            except ValueError:
                continue
    
    # Find the minimum affinity score
    if affinities:
        best_docking_score = min(affinities)
    else:
        best_docking_score = None
    return best_docking_score

# Define the paper lookup tool
@tool
def domain_knowledge(prompt):
    '''Useful for getting chemical intuition for the problem.
    This tool looks up design guidelines for molecules with higher binding afinity against WDR5 by looking through research papers.
    It also includes information on the paper citation or DOI.
    '''
    guide_lines = []
    for m in range(3):
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20)
        paper_file =f'../data/papers/Binding/WDR5/{m}.pdf'
        pages = PyPDFLoader(paper_file).load_and_split()
        sliced_pages = text_splitter.split_documents(pages)
        faiss_vectorstore = FAISS.from_documents(sliced_pages, OpenAIEmbeddings(model=Embedding_model))
        
        llm = ChatOpenAI(
            model_name='gpt-4',
            temperature=0.1,
        )
        g = dziner.RetrievalQABypassTokenLimit(faiss_vectorstore, RetrievalQA_prompt, llm)
        guide_lines.append(g)
    return " ".join(guide_lines)

# Define input function for human feedback
def get_input() -> str:
    return st.session_state.get("current_feedback", "")

# Setup human feedback tool
human_feedback = HumanInputRun(
    input_func=get_input,
    description="Use this tool to obtain feedback on the design of the new SMILES you suggested.",
    name='human_feedback'
)

# Combine all tools
tools = [domain_knowledge, predictGNINA_docking, human_feedback]
tool_names = [tool.name for tool in tools]
tool_desc = [tool.description for tool in tools]


# Define agent prompts
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
You have a final answer once you have as many new SMILES as you are asked.
       
"""

PREFIX = """You are a helpful Chemist AI assistant called dZiner. Answer to questions the best you can.
     Some questions may task you to make changes to a molecule based on some design guidelines
    and optimize the its property.
    You can look up these guidelines from the scientific literature for more information on what changes
    changes you can make. If you don't know, do not make up answers. Suggest one new SMILES ONLY.
    Explain changes you make to the molecule in details at each step but DO NOT put redundant information and make it short.
    """

SUFFIX = """If you're asked to design a molecule, start by:

1. Ask for the initial molecule to start with.

2. Researching Design Guidelines (Do this if you have not done already): Write a summary of the guidelines in a few bullet points, with citations (e.g., paper DOI) included for any design guidelines used.

3. Evaluating Initial SMILES, it it is provided, if not ask for it:

Binding Affinity: Assess the binding affinity of the initial SMILES.
Validity: Check the validity of the initial SMILES structure. 

4. Visualize the initial SMILES

5. Modifying SMILES:

Manual Adjustments: Based on the guidelines from Step 2, make manual changes to the SMILES. 
These changes may involve adding, removing, or replacing elements, functional groups, or rings in the molecule’s core.
Example: For a functionalized indole, try replacing the indole with structures like isoindole, indolizine, methylindoles,
 indolone, or methylenedioxybenzene.
Diverse Approaches: Apply different modifications that could enhance affinity without significantly increasing molecular size/weight.
Alignment with Guidelines: Justify in great details how each new SMILES aligns with the design guidelines, 
such as adding aromatic rings if suggested.

6. Evaluating Modified SMILES:

Binding Affinity: Assess the binding affinity of each new SMILES.
Validity: Ensure each new SMILES is valid.

7. Visualize the new SMILES.

 After each modification in step 6, visualize the new SMILES.


Based on the guidelines from step 2, redo steps 5-7 until you have as many new SMILES as you are asked.
Iteration Details:

Iteration 0: Use the initial input SMILES.

iteration results example:
"Iteration": 1,
      "SMILES": "CC(=O)OC1=CC=C(C=C1)C(=O)N2CCN(CC2)CC3=CC=CC=C3I",
      "Modification": The change made to the SMILES in the previous iteration and the detailed reasoning behind it.
      "Docking Score": -9.8,
      "SA Score": 2.01,
      "QED Score": 0.40,
      "Molecular Weight": 150,
      "Validity": "Valid"

Do not use the same SMILES you
 have evaluated before in a new iteration.

Final Steps:

Document All Changes: Even if reverted, all changes must be documented and reported. Report the tools used, and their citaions.
Avoid Redundancy: Do not repeat SMILES in the iteration history.
Begin with Problem Description:
Question: {input}
Thought: {agent_scratchpad}
"""
anthropic_api_key = ""

# Initialize the agent
agent_model = ChatAnthropic(
    model="claude-3-5-sonnet-20240620", 
    api_key=anthropic_api_key,  # Make sure this is set in your environment or config
    temperature=0.3,
    max_tokens=8192
)

agent = dZiner(
    tools,
    property="Binding affinity",
    model=agent_model,
    verbose=True,
    temp=0.3,
    suffix=SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    prefix=PREFIX,
    max_tokens=8192,
    max_iterations=120,
    n_design_iterations=25
).agent

# Store the agent in session state for reuse
if "agent" not in st.session_state:
    st.session_state.agent = agent



with st.sidebar:
    col1, col2, col3 = st.columns(3)
    col2.image('logo_transparent.png', width=150)
    st.title("dZiner v0.1")
    st.markdown(
        "**Chemist AI Agent for Rational Inverse Design of Materials.**"
    )
    api_key = st.text_input(
        "OpenAI API Key",
        placeholder="sk-...",
        help=f"['What is that?']({api_key_url})",
        type="password",
        value=os.environ["OPENAI_API_KEY"],
    )
    # os.environ["OPENAI_API_KEY"] = f"{api_key}"  #
    if len(api_key) != 51:
        st.warning("Please enter a valid OpenAI API key.", icon="⚠️")

  
    st.markdown(
        "⚠️ Note that this is only a demo version of the module. You can probably get much better\
                 results with your own customized prompts using the [software](https://github.com/mehradans92/dziner)."
    )



Embedding_model = 'text-embedding-3-large'  # You might want to verify this model
# RetrievalQA_prompt = "Design guidelines for molecules with higher binding affinity against WDR5"

RetrievalQA_prompt =  """What are the design guidelines, molecular substituents, and molecular interactions that give a molecule a
 strong binding affinity against WDR5 target? Ideally, provide a series of specialized molecular functional groups that interact
   with WDR5 and provide a good binding affinity that can be used to functionalize a more generic hit from a ligand discovery assay.
     Summarize your answer and cite paper by the title, DOI and the journal and year the document was published on.
"""

# RetrievalQA_prompt = """What are the design guidelines for scaffold hopping for making a molecule have a larger binding afinity against WDR5 target? 
#     This can be based on making changes to the functional groups or other changes to the molecule. Summarize your answer and cite paper
#     by the title, DOI and the journal and year the document was published on."""


def mol_to_img(smiles):
    '''
    Convert SMILES string to a molecule image with RDKit's customizable options.
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Return None if the SMILES is invalid
    
    # Use RDKit's MolDraw2D to customize drawing
    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 1000) 
    drawer.drawOptions().addAtomIndices = False  # Hide atom indices
    drawer.drawOptions().addStereoAnnotation = True  # Add stereo annotations
    drawer.drawOptions().highlightColour = (0.1, 0.2, 1.0)  # Custom highlight color for N/O
    
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    
    # Get the PNG binary data from the drawer
    img_bytes = drawer.GetDrawingText()
    
    # Convert to an image object for Streamlit to render
    img = Image.open(io.BytesIO(img_bytes))
    return img

def drug_chemical_feasibility(smiles: str):
    '''
    This tool inputs a SMILES of a drug candidate and outputs chemical feasibility, Synthetic Accessibility (SA),
    Quantentive drug-likeliness(QED) scores, molecular weight, and the SMILES.
    '''
    smiles = smiles.replace("\n", "")
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return "Invalid SMILES", 0, 0, 0, smiles

    # Calculate SA score
    sa_score = dziner.sascorer.calculateScore(mol)

    # Calculate molecular weight and QED score
    molecular_weight = Descriptors.MolWt(mol)
    qed_score = QED.qed(mol)
    
    return "Valid SMILES", sa_score, molecular_weight, qed_score, smiles

# Function to look up papers and extract guidelines
def lookup_papers(uploaded_file):
    '''Looks up design guidelines from an uploaded research paper PDF.
    It processes the PDF and extracts guidelines for molecular design.'''

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    try:
        # Initialize a list to store the guidelines
        guide_lines = []

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        
        # Load and split the temporary PDF file
        pages = PyPDFLoader(temp_file_path).load_and_split()
        sliced_pages = text_splitter.split_documents(pages)

        # Initialize FAISS vector store
        faiss_vectorstore = FAISS.from_documents(sliced_pages, OpenAIEmbeddings(model=Embedding_model))

        # Set up OpenAI's LLM model
        llm = ChatOpenAI(model_name='gpt-4o', temperature=0.1)
        
        # Use dziner's RetrievalQABypassTokenLimit to get design guidelines
        g = dziner.RetrievalQABypassTokenLimit(faiss_vectorstore, RetrievalQA_prompt, llm)
        paper_info_prompt = '''Give the title, journal name and year for publication of the paper without additional text.\
            example: "Design, synthesis, and initial evaluation, Bioorganic Chemistry (2021)" '''
        paper_info = dziner.RetrievalQABypassTokenLimit(faiss_vectorstore, paper_info_prompt, llm)

        guide_lines.append(g)

        # Return the guidelines as a string
        return "\n".join(guide_lines), paper_info
    
    finally:
        # Clean up: Delete the temporary file after processing
        os.remove(temp_file_path)

if "molecules_history" not in st.session_state:
    st.session_state["molecules_history"] = []


# Streamlit App



# Create two columns
with st.container(border=True):
    col1, col2 = st.columns([1, 1])  # Two equally sized columns

    # Left column for PDF uploads
    with col1:
        st.header("Upload Research Papers (PDF)")
        uploaded_files = st.file_uploader("Drag and drop PDFs here", type="pdf", accept_multiple_files=True)

        # Add a process button to trigger processing of uploaded PDFs
        process_button = st.button("Process PDFs")

        # Display uploaded file names with icons indicating processing status
        if uploaded_files:
            status_dict = {file.name: "📄" for file in uploaded_files}  # Initialize status with document icon
            status_placeholders = {file.name: st.empty() for file in uploaded_files}  # Placeholders for each file status
            for file_name, placeholder in status_placeholders.items():
                placeholder.markdown(f"{status_dict[file_name]} {file_name}")

    # If files are uploaded and the process button is clicked, process them and display in the right column
    with col2:
        st.header("Extracted Guidelines")

        if uploaded_files and process_button:
            # Process each uploaded PDF and extract guidelines
            for uploaded_file in uploaded_files:
                # Update status to show a rotating arrow while processing
                status_placeholders[uploaded_file.name].markdown(f"🔄 Processing {uploaded_file.name}...")

                with st.spinner(f"Processing {uploaded_file.name}..."):
                    guidelines, paper_info = lookup_papers(uploaded_file)
                
                # Display the extracted design guidelines after each PDF is processed
                st.text_area(f"Paper: {paper_info}", value=guidelines, height=300)

                # Update status to show checkmark after processing is complete
                status_placeholders[uploaded_file.name].markdown(f"✅ {uploaded_file.name} processed")
                
        elif uploaded_files and not process_button:
            st.info("Click the 'Process PDFs' button to start processing the uploaded files.")
        else:
            st.info("Please upload one or more PDF files to extract design guidelines.")

from streamlit_ketcher import st_ketcher
from concurrent.futures import ThreadPoolExecutor
import time

# Create two columns for the Molecule Editor and Design History, with fixed height alignment

# Default SMILES and molecule editor
DEFAULT_MOL = (
    r"C[N+]1=CC=C(/C2=C3\C=CC(=N3)/C(C3=CC=CC(C(N)=O)=C3)=C3/C=C/C(=C(\C4=CC=[N+](C)C=C4)C4=N/"
    "C(=C(/C5=CC=CC(C(N)=O)=C5)C5=CC=C2N5)C=C4)N3)C=C1"
)

if "input_locked" not in st.session_state:
    st.session_state.input_locked = False
if "initial_molecule" not in st.session_state:
    st.session_state["initial_molecule"] = None

# Left column (Molecule Editor using SMILES)
with st.container(border=True):
    st.header("Molecule Design Playground") 
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.write("Initial Molecule")
        sub_col1, sub_col2 = st.columns([3,1], gap='medium')
        with sub_col1:
            molecule = st.text_input("Initial Molecule", placeholder="Input SMILES" ,value=st.session_state.get("initial_molecule", ""), 
                                             label_visibility="collapsed", disabled=st.session_state.input_locked)
            # Lock the input if a valid molecule is entered
            if molecule and not st.session_state.input_locked:
                st.session_state.initial_molecule = molecule
                st.session_state.input_locked = True 
                st.rerun() 

        with sub_col2:
            if st.button("Restart"):
                st.session_state.initial_molecule = None  # Clear molecule in session state
                st.session_state.input_locked = False  # Unlock the input field
                st.session_state["molecules_history"] = []  # Clear history
                st.session_state.chat_history = [st.session_state.chat_history[0]]
                st.rerun() 
        
        if molecule:
            # Check validity and display result
            validity, sa_score, molecular_weight, qed_score, smiles = drug_chemical_feasibility(molecule)

            if validity == "Invalid SMILES":
                st.error(f"Error: {validity}. Please enter a valid SMILES.")
            else:
                smile_code = st_ketcher(molecule, height=700)
                st.markdown(f"Smile code: `{smile_code}")
                # Add the molecule and its data to the session state history
                new_entry = {
                    "smiles": smiles,
                    "SA": sa_score,
                    "MW": molecular_weight,
                    "QED": qed_score,
                    "iteration": len(st.session_state["molecules_history"])
                }
                st.session_state["molecules_history"].append(new_entry)

    # Right column (Chat Interface)
    with col2:
        if molecule:
            st.subheader("Current Design")
            with st.container(height=250):
                # Display each molecule one after another
                current_entry = st.session_state["molecules_history"][-1]
                # Create two sub-columns within the right column: 50% for the image, 50% for the information
                img_col, info_col = st.columns([0.8, 0.5], gap="small")

                # Generate the PNG image for the SMILES
                img_io = mol_to_img(current_entry["smiles"])

                # Display the molecule image in the left sub-column
                with img_col:
                    if img_io:
                        st.image(img_io, width=200)
                    else:
                        st.error("Unable to generate molecule image. Invalid SMILES string.")

                # Display molecular information in the right sub-column
                with info_col:
                    st.markdown(f"**Iteration {current_entry['iteration']}**")
                    st.write(f"**SMILES**: {current_entry['smiles']}")
                    st.write(f"**SA Score**: {current_entry['SA']:.3f}")
                    st.write(f"**Molecular Weight**: {current_entry['MW']:.2f}")
                    st.write(f"**QED Score**: {current_entry['QED']:.3f}")

            st.markdown("""
                <style>
                .stChatFloatingInputContainer {
                    bottom: 0;
                    background-color: var(--background-color);
                    z-index: 100;
                }
                
                [data-testid="stChatMessageContent"] {
                    padding: 1rem;
                    margin-bottom: 0.5rem;
                }
                
                .chat-messages {
                    overflow-y: auto;
                }
                </style>
            """, unsafe_allow_html=True)

            st.subheader("dZiner")

            # Define assistant avatar
            assistant_avatar = "chat_icon.png"

            # Main chat container with fixed height
            with st.container(height=500):
                # Initialize chat history and flag for first message if it doesn't exist
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                    
                # Use a flag to track if the initial message has been added
                if "initial_message_added" not in st.session_state:
                    st.session_state.initial_message_added = False

                # Add the initial message from Dziner only once
                if not st.session_state.initial_message_added:
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": "Hi! I'm Dziner. How can I assist you with your molecule design?"
                    })
                    st.session_state.initial_message_added = True

                # Add spacing at top
                st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)

                # Display chat history
                for message in st.session_state.chat_history:
                    if message["role"] == "assistant":
                        with st.chat_message("assistant", avatar=assistant_avatar):
                            st.write(message["content"])
                    else:
                        with st.chat_message("user"):
                            st.write(message["content"])

                # Add padding at the bottom
                st.markdown('<div style="height: 80px;"></div>', unsafe_allow_html=True)

            # Chat input
            if prompt := st.chat_input("Ask about your molecule..."):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                current_entry = st.session_state["molecules_history"][-1]
                input_data = {
                    "input": prompt,
                    "prompt": RetrievalQA_prompt,
                    "tools": tools,
                    "tool_names": tool_names,
                    "tool_desc": tool_desc
                }

                def response_generator():
                    f = io.StringIO()
                    with redirect_stdout(f):
                        agent_response = agent.invoke(input_data)
                        
                    # Get the captured output
                    output = f.getvalue()
                    
                    # Process each line
                    for line in output.split('\n'):
                        if line.strip():
                            # Prepare the formatted line based on its type
                            if line.startswith('Thought:'):
                                formatted_line = f"**🤔 {line}**\n"
                            elif line.startswith('Action:'):
                                formatted_line = f"**⚡ {line}**\n"
                            elif line.startswith('Action Input:'):
                                formatted_line = f"**📥 {line}**\n"
                            elif line.startswith('Observation:'):
                                formatted_line = f"**👁️ {line}**\n"
                            elif line.startswith('Final Answer:'):
                                formatted_line = f"**✅ {line}**\n"
                            else:
                                formatted_line = f"{line}\n"
                            
                            # Stream each character with a small delay
                            for char in formatted_line:
                                yield char
                                time.sleep(0.02)  # Adjust this value to control typing speed
                        
                        # Add a line break between different sections
                        yield "\n"
                        time.sleep(0.1)  # Slightly longer pause between lines

                # Stream the response
                with st.chat_message("assistant", avatar=assistant_avatar):
                    response = st.write_stream(response_generator())
                    
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                        


col1, col2, col3 = st.columns([1, 2, 1], gap="medium")
with col2:
    if molecule:
        st.header("Design History")
        with st.container(height=500):
            # Display each molecule one after another
            for entry in st.session_state["molecules_history"]:
                # Create two sub-columns within the right column: 50% for the image, 50% for the information
                img_col, info_col = st.columns([0.8, 0.5], gap="small")

                # Generate the PNG image for the SMILES
                img_io = mol_to_img(entry["smiles"])

                # Display the molecule image in the left sub-column
                with img_col:
                    if img_io:
                        st.image(img_io, width=300)
                    else:
                        st.error("Unable to generate molecule image. Invalid SMILES string.")

                # Display molecular information in the right sub-column
                with info_col:
                    st.markdown(f"**Iteration {entry['iteration']}**")
                    st.write(f"**SMILES**: {entry['smiles']}")
                    st.write(f"**SA Score**: {entry['SA']:.3f}")
                    st.write(f"**Molecular Weight**: {entry['MW']:.2f}")
                    st.write(f"**QED Score**: {entry['QED']:.3f}")
                st.write("---")  # Separate each molecule iteration



        
if __name__ == "__main__":
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/mehradansari">Mehrad Ansari</a></h6>',
            unsafe_allow_html=True,
        )


