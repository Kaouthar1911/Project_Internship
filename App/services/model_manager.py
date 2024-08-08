import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import ServiceContext, Settings
from config import Config

class ModelManager:
    def __init__(self):
        self.embed_model = HuggingFaceEmbedding(model_name=Config.EMBED_MODEL_NAME)
        self.llm = Groq(model=Config.LLM_MODEL, api_key=os.getenv('GROQ_API_KEY'))
        self.text_splitter = SentenceSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        Settings.text_splitter = self.text_splitter
        self.service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model)