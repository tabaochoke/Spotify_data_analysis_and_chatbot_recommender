# from langchain.agents.agent_types import AgentType
# from langchain.chat_models import ChatOpenAI
# from langchain.llms.openai import OpenAI
# from langchain_experimental.agents.agent_toolkits import create_python_agent
# from langchain_experimental.tools import PythonREPLTool
# from langchain.llms import HuggingFaceHub
# import os
# from getpass import getpass
# #
# from getpass import getpass
# HUGGINGFACEHUB_API_TOKEN = getpass()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


# HUGGINGFACEHUB_API_TOKEN = getpass()

# import os
# from configparser import ConfigParser
# env_config = ConfigParser()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


# # Retrieve the cohere api key from the environmental variables
# def read_config(parser: ConfigParser, location: str) -> None:
#  assert parser.read(location), f"Could not read config {location}"
# #
# CONFIG_FILE = os.path.join(".", ".env")
# read_config(env_config, CONFIG_FILE)
# api_key = env_config.get("cohere", "api_key").strip()
# os.environ["COHERE_API_KEY"] = api_key
# api_key = env_config.get("hgface", "api_key").strip()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key



# repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
# llm = HuggingFaceHub(
# repo_id=repo_id, model_kwargs={"temperature": 0.75, "max_length": 300}
# )
    
# agent_executor = create_python_agent(
#     llm=llm,
#     tool=PythonREPLTool(),
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# )

# agent_executor.run("What is the 10th fibonacci number?")


# import requests
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
# img_path = './image/nike_full_black.png' 
# raw_image = Image.open(img_path).convert('RGB')

# # conditional image captioning
# text = "a person is wearing "
# inputs = processor(raw_image, text, return_tensors="pt")
# print ("dss")
# out = model.generate(**inputs , max_new_tokens=100)
# print(processor.decode(out[0], skip_special_tokens=True  ))


# from langchain.text_splitter import CharacterTextSplitter

# state_of_the_union = "Your long text here. what do you want. hehe he. "

# text_splitter = CharacterTextSplitter(
#     separator="r",
#     chunk_size=1,
#     chunk_overlap=1,
#     length_function=len,
#     is_separator_regex=False,
# )
# texts = text_splitter.create_documents([state_of_the_union])
# print(texts)

# from sentence_transformers import SentenceTransformer , util
# import numpy as np

# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


# user_input = ["I am going to gym, what do you think will suit me"]

# sentence_2_compare = ['I want go to virtual fitting room' , 
#                       'Recommend some outfit for me with my request',
#                       'Base on my image, recommend outfit for my clothes',
                      
#                       ""]

# user_embedding = model.encode (user_input)
# scores = []
# for sentence in sentence_2_compare :
#     sentence_embedding = model.encode (sentence)
#     print ( util.pytorch_cos_sim(user_embedding, sentence_embedding ))
#     scores.append ( util.pytorch_cos_sim(user_embedding, sentence_embedding )[0][0].item() )
    
# print (scores)
# print (np.argmax (np.array (scores)) )

import torch
print ( torch.cuda.is_available ())
