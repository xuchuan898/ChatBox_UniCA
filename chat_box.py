# Import required libraries
import argparse
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


def prepare_data():
    #persist_directory = './docs/chroma/'
        # Create embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
            #model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # load docs (adjust path / loader)
    #loader = PyPDFLoader("./docs/chroma/MASTER_IA_Information.pdf")      # or PyPDFLoader("file.pdf")
    loader = TextLoader("./docs/chroma/master.md", encoding="utf-8")

    docs = loader.load()

    #linkloader1 = WebBaseLoader("https://univ-cotedazur.fr/formation/offre-de-formation/parcours-de-master/master-informatique-parcours-intelligence-artificielle#programme")
    #linkloader2 = WebBaseLoader(" https://univ-cotedazur.fr/formation/offre-de-formation/parcours-de-master/master-informatique-parcours-systemes-logiciels-et-calculs-distribues")
    #linkloader3 = WebBaseLoader("https://upinfo.univ-cotedazur.fr/master/liste/")
    #linkdoc = linkloader3.load()  

    docs = docs
    # split into chunks
    # splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # texts = splitter.split_documents(docs)

    # define markdown headers
    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]

    # markdown splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    texts = []

    for doc in docs:
        splits = markdown_splitter.split_text(doc.page_content)
        for s in splits:
            header_context = " ".join([
                s.metadata.get("H1", ""),
                s.metadata.get("H2", ""),
                s.metadata.get("H3", "")
            ])

            texts.append(
                Document(
                    page_content=header_context + "\n" + s.page_content,
                    metadata=s.metadata
                )
         )   

    token_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=120
    )

    texts = token_splitter.split_documents(texts)

    # create vector DB
    vectordb = Chroma.from_documents(
        texts,
        embedding=embeddings,
        #persist_directory=persist_directory
    )

    return vectordb 

def chatbox(vectordb):
    llm = ChatOllama(
        
        model="gemma3:1b",
            
        validate_model_on_init=True,
            
        temperature=0.8,
            
        num_predict=256,
        
    # other params ...
    )

    prompt = ChatPromptTemplate.from_template("""
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use the context to answer concisely. Keep the answer as concise as possible. Always say "Merci pour votre question!" at the end of the answer. Please answer the question in the langugae used by the question
        {context}

        Question: {input}
        Answer:""")

    # Run chain

    retriever = vectordb.as_retriever(
        search_type="mmr",  # Use MMR to diversify results
        search_kwargs={"k": 10, "fetch_k": 15}  # Retrieve 10 docs, consider 15 initially
    )
    print(vectordb._collection.count())
    for d in retriever.invoke("Quels sont les cours en semestre 1?"):
        print("-----")
        print(d.page_content[:300])

    # Build the document combination chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain (this replaces RetrievalQA)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbox script.")
    parser.add_argument("-q", "--question", type=str, required=True, help="Question to ask.")
    return parser.parse_args()


def main() -> None:
    os.environ["USER_AGENT"] = "MyChatBot/1.0"
    qa_chain=chatbox(prepare_data())
    print("Chat started. Type 'exit' to quit.")
    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue
        result = qa_chain.invoke({"input": user_input})
        print_result = result["answer"]
        print(f"--Question received: {user_input}")
        print(f"**Bot Answer**: {print_result} \\")
    


if __name__ == "__main__":
    main()