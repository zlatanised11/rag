import os
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()

class RAGChat:
    def __init__(self, markdown_directory='./'):
        self.markdown_directory = markdown_directory
        self.setup_rag_system()
        
    def load_documents(self):
        docs = []
        for filename in os.listdir(self.markdown_directory):
            if filename.endswith('.md'):
                filepath = os.path.join(self.markdown_directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                docs.append(Document(page_content=text, metadata={"source": filepath}))
        return docs
        
    def setup_rag_system(self):
        # Load and split documents
        docs = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128
        )
        split_docs = text_splitter.split_documents(docs)
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        # Setup retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Setup LLM and compression
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        
        # Create prompt template
        system_prompt = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, output NA. Keep the answer very concise with as few words as possible, since it's a factual question.
        In essence, give the answer if you know it and NA otherwise. Do not give any further explanation if you are not sure of the answer.
        Analyse the question carefully as it is possible for some reasoning to be involved while answering. However, do not output your reasoning. 
        If you know the correct answer, then output it. Otherwise give output as NA.

        {context}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)
    
    def get_answer(self, question):
        answer = self.rag_chain.invoke({"input": question})
        return answer['answer']
