import os
os.environ["OPENAI_API_KEY"] = "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"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


if __name__ == "__main__":
    # Load the PDF file
    loader = PyPDFLoader("max-muscle-at-50.pdf")
    documents = loader.load()
    text_splitter= CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,    # Adjust chunk size as needed   
        chunk_overlap=30,  # Adjust overlap as needed
    )
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index_react")
    
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Show me the table for Monday workout"})
    print(res["answer"])

    # # Print the number of documents loaded
    # print(f"Number of documents loaded: {len(documents)}")

    # # Print the first document's content
    # if documents:
    #     print(f"First document content: {documents[0].page_content[:500]}...")  # Print first 500 characters
    # else:
    #     print("No documents were loaded.")

