import os
import shutil
from llama_index.core import SimpleDirectoryReader, Document
from config import Config
from .index_manager import IndexManager

class AdminManager:
    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager

    def add_new_document(self, file_path: str):
        file_path = file_path.strip('"')
        if not os.path.exists(file_path):
            return f"File {file_path} does not exist."
        
        filename = os.path.basename(file_path)
        destination = os.path.join(Config.DATA_DIR, filename)
        shutil.copy2(file_path, destination)
        
        new_docs = SimpleDirectoryReader(input_files=[destination]).load_data()
        self.index_manager.update_index(new_docs)
        return f"Document {filename} added successfully."

    def update_document(self, filename: str, new_content: str):
        file_path = os.path.join(Config.DATA_DIR, filename)
        if not os.path.exists(file_path):
            return f"Document {filename} not found."
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        new_doc = Document(text=new_content, metadata={"file_name": filename})
        self.index_manager.update_index([new_doc])
        return f"Document {filename} updated successfully."

    def delete_document(self, filename: str):
        file_path = os.path.join(Config.DATA_DIR, filename)
        if not os.path.exists(file_path):
            return f"Document {filename} not found."
        
        os.remove(file_path)
        # Note: Deleting from the index is more complex and may require rebuilding the index
        return f"Document {filename} deleted successfully. You may need to rebuild the index."

    def list_documents(self):
        return os.listdir(Config.DATA_DIR)