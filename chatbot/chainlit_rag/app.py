#import required libraries
from image import *
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter , CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQAWithSourcesChain,RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
 SystemMessagePromptTemplate,
 HumanMessagePromptTemplate)
from langchain.llms import HuggingFaceHub
from langchain.llms import CTransformers
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain ,ConversationalRetrievalChain,StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from chainlit import run_sync
from cohere import Client
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
import PyPDF2
from io import BytesIO
from getpass import getpass
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import torch
HUGGINGFACEHUB_API_TOKEN = getpass()
from langchain.llms import HuggingFaceTextGenInference
import os
from configparser import ConfigParser
env_config = ConfigParser()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


# Retrieve the cohere api key from the environmental variables
def read_config(parser: ConfigParser, location: str) -> None:
    assert parser.read(location), f"Could not read config {location}"
#
CONFIG_FILE = os.path.join(".", ".env")
read_config(env_config, CONFIG_FILE)
api_key = env_config.get("cohere", "api_key").strip()
os.environ["COHERE_API_KEY"] = api_key
api_key = env_config.get("hgface", "api_key").strip()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

text_splitter = CharacterTextSplitter(
    separator="Song",
    chunk_size=1,
    chunk_overlap=1,
    length_function=len,
    is_separator_regex=False,
)

# system_template = """
# <s> [INST] You are a music assistant. Use user information to recommend suitable song for them. [/INST]
# """
# messages = [SystemMessagePromptTemplate.from_template(system_template),HumanMessagePromptTemplate.from_template("{question}"),]
# prompt = ChatPromptTemplate.from_messages(messages)
# chain_type_kwargs = {"prompt": prompt}

import qdrant_client

from huggingface_hub import login
access_token_write = "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"
login(token = access_token_write)



#Decorator to react to the user websocket connection event.
@cl.on_chat_start
async def init():
    msg = cl.Message(content=f"Hello I am Baymax, Nice to meet you")
    file_path = 'example.txt'
    texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            texts.append (line)

    # print ("length_Text" , len (texts))
    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    # Create a Chroma vector store
    model_id = "BAAI/bge-small-en-v1.5"
    # model_id = "BAAI/bge-large-en-v1.5"
    # model_id = "WhereIsAI/UAE-Large-V1"
    embeddings = HuggingFaceBgeEmbeddings(model_name= model_id,model_kwargs = {"device":"cpu"})
    cl.user_session.set("embeddings", embeddings)
    #
    # Store the embeddings in the user session
    docsearch = await cl.make_async(Qdrant.from_texts)(
    texts, embeddings,location=":memory:", metadatas=metadatas
    )
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    #Hybrid Search
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k=5
    qdrant_retriever = docsearch.as_retriever(search_kwargs={"k":5})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,qdrant_retriever],
    weights=[0.5,0.5])
    #Cohere Reranker 
    compressor = CohereRerank(client=Client(api_key=os.getenv("COHERE_API_KEY")),user_agent='langchain')
    #
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
    base_retriever=ensemble_retriever,
    )
    # Create a chain that uses the Chroma vector store
    repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 5000})

    template = (
            "You are a helpful AI music assistant that recommend song for user.Using only song provided, do not make up any song. Give the user genius URL of the song after you recommend "
            "Combine the chat history and follow up question into "
            "a standalone question. Chat History: {chat_history}"
            "Follow up question: {question} "
        )
    prompt = PromptTemplate.from_template(template)
    print ("prompt" , prompt)
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
        
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        condense_question_prompt = prompt,
        chain_type="stuff",
        retriever=compression_retriever,
        memory=memory,
        return_source_documents=True,
        # combine_docs_chain=combine_docs_chain,
    )
    
    # Save the metadata and texts in the user session
    # cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)
    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()
    #store the chain as long as the user session is active
    cl.user_session.set("chain", chain)
    
    
@cl.on_message
async def process_response(res:cl.Message):
    # special_word = ['png','jpeg']
    # retrieve the retrieval chain initialized for the current session
    chain = cl.user_session.get("chain") 
    # Chainlit callback handler
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    print("in retrieval QA")
    #res.content to extract the content from chainlit.message.Message
    print(f"res : {res.content}")
    response = await chain.acall(res.content, callbacks=[cb])

    # response = await chain.acall(input, callbacks=[cb])
    
    answer = response["answer"]
    print(f"response: {answer}")
     #quan trong dung de lay cau tra loi

    #Retrieve source document
    sources = response["source_documents"]
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    # all_sources = [m["source"] for m in metadatas]
    # texts = cl.user_session.get("texts")
    if sources:
        found_sources = []
        # Add the sources to the message
        for source in sources:
            print(source.metadata)
            try :
                source_name = source.metadata["source"]
            except :
                source_name = ""
            # Get the index of the source
            text = source.page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()




# async def show_image(res):
#     contents = res.split(' ')[-1]

#     path = './image/'+contents
#     elements = [
#         cl.Image(name="image1", display="inline", path=path)
#     ]
#     new_response = 'Here is my recommendation: '
#     await cl.Message(content=new_response, elements=elements).send()
# @cl.action_callback("rating_button")
# async def on_action(action):
#     await action.remove()
#     await cl.Message(content=f"Your outfit score based on my provided knowledge: 10").send()

# @cl.action_callback("inter_virtual_fittingroom")
# async def on_action(action):
#     await action.remove()
#     await cl.Message(content=f"Now you can try on your outfit! Have a nice experimence").send()

# async def show_button():
#     # Sending an action button within a chatbot message
#     actions = [
#         cl.Action(name="rating_button", value="example_value", description="Click me!"),
#         cl.Action(name="enter_virtual_fittingroom", value="example_value", description="Click me!")
#     ]

#     await cl.Message(content="You can use these function for more experimence:", actions=actions).send()
