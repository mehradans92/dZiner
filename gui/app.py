import streamlit as st
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

Embedding_model = 'text-embedding-3-large'  # You might want to verify this model
RetrievalQA_prompt = "Design guidelines for molecules with higher binding affinity against WDR5"


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
        llm = ChatOpenAI(model_name='gpt-4', temperature=0.1)
        
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


# Streamlit App
st.set_page_config(layout="wide")
st.title("`dZiner v0.1`")

# Create two columns
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

# Create two columns for the Molecule Editor and Design History, with fixed height alignment
col1, col2 = st.columns([2, 1], gap="medium")  # Left column for SMILES input and molecule editor, right column for displaying the PNG and information

# Default SMILES and molecule editor using st_ketcher
DEFAULT_MOL = (
    r"C[N+]1=CC=C(/C2=C3\C=CC(=N3)/C(C3=CC=CC(C(N)=O)=C3)=C3/C=C/C(=C(\C4=CC=[N+](C)"
    "C=C4)C4=N/C(=C(/C5=CC=CC(C(N)=O)=C5)C5=CC=C2N5)C=C4)N3)C=C1"
)

# Left column (Molecule Editor using st_ketcher)
with col1:
    st.header("Molecule Design Playground")
    molecule = st.text_input("Initial Molecule", value=DEFAULT_MOL)

    # Check validity and display result
    validity, sa_score, molecular_weight, qed_score, smiles = drug_chemical_feasibility(molecule)

    if validity == "Invalid SMILES":
        st.error(f"Error: {validity}. Please enter a valid SMILES.")

    # Ketcher component to visualize the molecule
    smile_code = st_ketcher(molecule)
    st.markdown(f"Smile code: ``{smile_code}``")

# Right column (Design History with aligned height)
with col2:
    st.header("Design history")

    # Create two sub-columns within the right column for alignment
    img_col, info_col = st.columns([0.5, 0.5])  # 50% for image, 50% for information

    # Generate and display the PNG image using the SMILES captured from st_ketcher
    img_io = mol_to_img(molecule)  # Convert SMILES to PNG
    
    # Display image in the left part of the right column
    with img_col:
        if img_io:
            st.image(img_io, caption="Iteration 0 Molecule", use_column_width=True)
        else:
            st.error("Unable to generate molecule image. Invalid SMILES string.")
    
    # Display molecular info (SA score, MW, QED) in the right part of the right column
    with info_col:
        validity, sa_score, molecular_weight, qed_score, smiles = drug_chemical_feasibility(molecule)
        if validity == "Valid SMILES":
            st.write(f"**SA Score**: {sa_score:.3f}")
            st.write(f"**Molecular Weight**: {molecular_weight:.2f}")
            st.write(f"**QED Score**: {qed_score:.3f}")
        else:
            st.error("Invalid SMILES for calculating properties.")

# Additional alignment to ensure both columns have similar height
st.write("---")


    # Additional iterations can be appended here as the user modifies the molecule

st.write("---")
