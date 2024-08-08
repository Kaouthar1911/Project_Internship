import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, Document
from config import Config
from .model_manager import ModelManager
from typing import List, Optional

class IndexManager:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.documents = self.load_documents()
        self.index = self.load_index_from_storage()

    def load_documents(self):
        return SimpleDirectoryReader(input_dir=Config.DATA_DIR).load_data()

    def load_index_from_storage(self):
        if os.path.exists(Config.PERSIST_DIR):
            storage_context = StorageContext.from_defaults(persist_dir=Config.PERSIST_DIR)
            return load_index_from_storage(storage_context, service_context=self.model_manager.service_context)
        else:
            return self.create_new_index()

    def create_new_index(self):
        index = VectorStoreIndex.from_documents(
            self.documents, 
            embed_model=self.model_manager.embed_model, 
            transformations=[self.model_manager.text_splitter],
        )
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
        return index

    def update_index(self, new_documents: List[Document]):
        self.documents.extend(new_documents)
        for doc in new_documents:
            self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=Config.PERSIST_DIR)