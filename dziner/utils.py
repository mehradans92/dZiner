from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from openai import OpenAIError

def RetrievalQABypassTokenLimit(vector_store, RetrievalQA_prompt, llm, k=10, fetch_k=50, min_k=2, chain_type="stuff"):
    while k >= min_k:
        try:
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k},
            )
            qa_chain = RetrievalQA.from_chain_type( 
                llm=llm,
                chain_type=chain_type,
                retriever=retriever,
                memory=ConversationBufferMemory(
                        ))
            
            # Check to see if we hit the token limit
            result = qa_chain.run({"query": RetrievalQA_prompt})
            return result  # If successful, return the result and exit the function

        except OpenAIError as e:
            # Check if it's a token limit error
            print(e)
            if 'maximum context length' in str(e):
                print(f"\nk={k} results hitting the token limit. Reducing k and retrying...\n")
                k -= 1
            else:
                # Re-raise other OpenAI errors
                raise e