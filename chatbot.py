import os
import sys
from pinecone import Pinecone
from langchain.llms import Replicate
#import replicate
#from langchain.vectorstores import Pinecone as lang_pine
from langchain_pinecone import Pinecone as lang_pine
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain import PromptTemplate
#from sentence_transformers import SentenceTransformer, util
import numpy as np
#similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def pdf_chatbot(ques, pdf_bytes):
    import io
    # Load and preprocess the PDF document

    # Replicate API token
    os.environ['REPLICATE_API_TOKEN'] = 'r8_eZmofbYtsCHBbaHQKq9Gs8zk3kB0wlz1Ao01S' #replicate API key 
    os.environ['PINECONE_API_KEY'] = '68dda7f6-43ac-4340-8461-56396ae32cd1'
    # Initialize Pinecone
    pc = Pinecone(api_key='68dda7f6-43ac-4340-8461-56396ae32cd1', environment='gcp-starter') #your index environment
    #Pinecone(api_key='68dda7f6-43ac-4340-8461-56396ae32cd1')#'YOUR PINECONE API HERE')

    # Use HuggingFace embeddings for transforming text into numerical vectors
    embeddings = HuggingFaceEmbeddings()

    # Initialize Replicate Llama2 Model
    llm = Replicate(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        input={"temperature": 0.75, "max_length": 3000}
    )

    prompt_template = """Use the following pieces of context to answer the question at the end.
            If you don't know the answer, please think rationally answer from your own knowledge base
            
    please provide correct context document
            
    Question: what is blackhole?
    """


    documents = ""
    reader = None
    with io.BytesIO(pdf_bytes.getvalue()) as open_pdf_file:
        reader = PdfReader(open_pdf_file)
        num_pages = len(reader.pages)
        print(num_pages)
        for i in range(0,len(reader.pages)):
            page1 = reader.pages[i]
            text = page1.extract_text()
            documents +=  f" ### Page {i}:\n\n" + text
        
    #documents = PdfFileReader(pdf_bytes.getvalue()) #pdf that need to be read
    #documents = loader.load()
 
    # Split the documents into smaller chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0,separator= " ### ")
    texts = text_splitter.create_documents(documents)

    # Set up the Pinecone vector database
    index_name = 'chatbot'#"NAME OF YOUR DATABASE(PINECONE INDEX THAT YOU CREATED)"
    index = pc.Index(index_name)
    vectordb = lang_pine.from_documents(texts, embeddings, index_name=index_name)



    PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

    chain_type_kwargs = {"prompt": PROMPT}

    # Set up the Conversational Retrieval Chain
    #qa_chain = RetrievalQA.from_chain_type(llm = 
        #llm,chain_type="stuff",retriever=
        #vectordb.as_retriever(search_kwargs={'k': 3}),chain_type_kwargs=chain_type_kwargs,
        #return_source_documents=False
    #)

    qa_chain = ConversationalRetrievalChain.from_llm(llm = 
        llm,retriever=
        vectordb.as_retriever(search_kwargs={'k': 3}),chain_type_kwargs=chain_type_kwargs,
        return_source_documents=False
    )


    # Start chatting with the chatbot
    chat_history = []
    #while True:
    query = ques
    #if query.lower() in ["exit", "quit", "q"]:
        #print('Exiting')
            #sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    #sim_results = vectordb.similarity_search(query,k=3) 
    print(result)
    #print("similarity results")
    #print(sim_results[0].page_content)
    return result 
        
    '''
    if (result['result']==''):
        print("I dont find the answer in pdf")
    else:
        print('Answer: '+result['result']+'\n')
    chat_history.append((query, result))
    '''
