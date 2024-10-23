import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import dziner
import tempfile
import os

Embedding_model = 'text-embedding-3-large'  # You might want to verify this model
RetrievalQA_prompt = "Design guidelines for molecules with higher binding affinity against WDR5"

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

# Default molecule for Ketcher
DEFAULT_MOL = (
    r"C[N+]1=CC=C(/C2=C3\C=CC(=N3)/C(C3=CC=CC(C(N)=O)=C3)=C3/C=C/C(=C(\C4=CC=[N+]"
    "(C)C=C4)C4=N/C(=C(/C5=CC=CC(C(N)=O)=C5)C5=CC=C2N5)C=C4)N3)C=C1"
)

from streamlit_ketcher import st_ketcher
molecule = st.text_input("Molecule", DEFAULT_MOL)
smile_code = st_ketcher(molecule)
st.markdown(f"Smile code: ``{smile_code}``")

st.write("---")
