import pytest
from unittest.mock import patch
from fake_llm import FakeChatModel
from langchain.schema import Document, ChatResult, ChatGeneration, AIMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAIError
from dziner import RetrievalQABypassTokenLimit

def test_RetrievalQABypassTokenLimit():
    # Create simple documents
    documents = [
        Document(page_content="Document about AI."),
        Document(page_content="Document about machine learning."),
        Document(page_content="Document about language models."),
    ]

    # Initialize OpenAIEmbeddings and FAISS vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    # Define a FakeChatModel that simulates LLM responses
    responses = ["This is a response within token limit.", 
                 "This is another response within token limit."]
    llm = FakeChatModel(responses=responses)

    # Set up the RetrievalQA prompt
    RetrievalQA_prompt = "What are these documents about?"

    # Call the function and capture the result
    result = RetrievalQABypassTokenLimit(
        vector_store=vector_store,
        RetrievalQA_prompt=RetrievalQA_prompt,
        llm=llm,
        k=3,
        fetch_k=5,
        min_k=2,
        chain_type="stuff"
    )

    # Assertions to check the result
    assert result is not None
    assert isinstance(result, str)
    assert result == "This is a response within token limit."


@patch.object(FakeChatModel, '_generate')
def test_RetrievalQABypassTokenLimit_token_limit_exceeded(mock_generate):
    # Create large documents to trigger token limit reduction
    large_text = "This is a large document content. " * 500
    documents = [Document(page_content=large_text) for _ in range(3)]

    # Initialize OpenAIEmbeddings and FAISS vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    # Define a FakeChatModel that raises OpenAIError once and then succeeds
    responses = [
        "This is a response that exceeds the token limit.",
        "Another response within token limit."
    ]
    llm = FakeChatModel(responses=responses)

    # Mock the _generate method to raise an OpenAIError on the first call and succeed afterward
    mock_generate.side_effect = [
        OpenAIError("maximum context length exceeded"),
        ChatResult(generations=[ChatGeneration(message=AIMessage(content="This is another response within token limit."))])
    ]

    RetrievalQA_prompt = "Summarize these large documents."

    # Capture stdout to check print statements
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        result = RetrievalQABypassTokenLimit(
            vector_store=vector_store,
            RetrievalQA_prompt=RetrievalQA_prompt,
            llm=llm,
            k=3,
            fetch_k=5,
            min_k=2,
            chain_type="stuff"
        )
    output = f.getvalue()

    # Check if the function reduced k due to token limit and then succeeded
    assert result is not None
    assert isinstance(result, str)
    assert "k=3 results hitting the token limit. Reducing k and retrying..." in output
    assert result == "This is another response within token limit."

    print("Test Output:\n", output)
    print("Test Result:", result)
