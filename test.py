import os
from llama_index.core.llms import ChatMessage, MessageRole
import logging
from typing import List, Optional
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, StorageContext, 
    load_index_from_storage, ServiceContext, Settings, Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.prompts import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
import shutil
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
import docx2txt
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import re

class Config:
    PERSIST_DIR = "./stocker"
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama3-70b-8192"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 20
    DATA_DIR = "./data"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.embed_model = HuggingFaceEmbedding(model_name=Config.EMBED_MODEL_NAME)
        self.llm = Groq(model=Config.LLM_MODEL, api_key=os.getenv('GROQ_API_KEY'))
        Settings.text_splitter = SentenceSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        self.service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model)

class TextOnlyReader:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir

    def load_data(self) -> List[Document]:
        all_docs = []
        for filename in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, filename)
            if filename.endswith('.txt'):
                text = self.read_txt(file_path)
                num_pages = 1  # Assuming one page for text files
            elif filename.endswith('.docx'):
                text = self.read_docx(file_path)
                num_pages = 1  # You might want to implement a page count for docx files
            elif filename.endswith('.pdf'):
                text, num_pages = self.read_pdf(file_path)
            else:
                Config.logger.warning(f"Skipping unsupported file type: {filename}")
                continue
            
            doc = Document(text=text, metadata={"file_name": filename, "num_pages": num_pages})
            all_docs.append(doc)
            Config.logger.info(f"Document {filename} loaded with {num_pages} pages")
        return all_docs

    def read_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def read_docx(self, file_path: str) -> str:
        return docx2txt.process(file_path)

    def read_pdf(self, file_path: str) -> tuple[str, int]:  
        Config.logger.info(f"Début de l'extraction du texte pour le fichier : {file_path}")
        text = ""
        num_pages = 0
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                Config.logger.info(f"Nombre total de pages dans le PDF : {num_pages}")

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text() or ""
                    text += page_text
                    Config.logger.info(f"Page {page_num}/{num_pages} extraite. Longueur du texte : {len(page_text)} caractères")

            total_length = len(text)
            Config.logger.info(f"Extraction terminée. Longueur totale du texte : {total_length} caractères")
            
            if total_length < 100:
                Config.logger.warning(f"Le texte extrait est très court ({total_length} caractères). Utilisation de l'OCR.")
                text = self.ocr_pdf(file_path)
            
            Config.logger.info(f"Aperçu du début du texte : {text[:200]}...")
            Config.logger.info(f"Aperçu de la fin du texte : ...{text[-200:]}")

        except Exception as e:
            Config.logger.error(f"Erreur lors de l'extraction du texte du PDF : {str(e)}")
            return "", 0

        return text, num_pages

    def ocr_pdf(self, file_path: str) -> str:
        images = convert_from_path(file_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image, lang='fra')
        return text

class IndexManager:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.index = self.load_or_create_index()

    def load_or_create_index(self):
        if os.path.exists(Config.PERSIST_DIR):
            storage_context = StorageContext.from_defaults(persist_dir=Config.PERSIST_DIR)
            return load_index_from_storage(storage_context, service_context=self.model_manager.service_context)
        
        reader = TextOnlyReader(input_dir=Config.DATA_DIR)
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=self.model_manager.service_context)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
        return index

    def update_index(self, new_documents: List[Document]):
        for doc in new_documents:
            self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=Config.PERSIST_DIR)

class PromptManager:
    @staticmethod
    def get_prompt_template():
        template = """
        Vous êtes un assistant compétent et précis, spécialisé dans le support client de PORTNET S.A. 
        Votre objectif est de fournir des réponses précises et pertinentes aux questions des utilisateurs.

        Instructions :
        1. Analysez soigneusement le contexte et la question.
        2. Identifiez les informations clés pertinentes à la question.
        3. Formulez une réponse claire et concise, en synthétisant les informations du contexte.
        4. Assurez-vous que votre réponse est directement liée à la question posée.
        5. Si le contexte ne fournit pas assez d'informations, indiquez-le clairement.
        6. Si l'information demandée existe dans plusieurs sources, posez des questions précises pour choisir la source la plus pertinente.
        7. Si on vous demande qui vous êtes, dites que vous êtes le chatbot de PortNet.

        Contexte : {context_str}
        Question: {query_str}
        Réponse:
        """
        return PromptTemplate(template=template)

class ChatbotEngine:
    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        self.retriever = VectorIndexRetriever(
            index=self.index_manager.index,
            similarity_top_k=25,
        )
        self.chat_engine = self.initialize_chat_engine()

    def initialize_chat_engine(self):
        prompt_template = PromptManager.get_prompt_template()
        response_synthesizer = get_response_synthesizer(
            response_mode="simple_summarize",
            service_context=self.index_manager.model_manager.service_context,
            text_qa_template=prompt_template,
            use_async=False,
            streaming=False,
        )
        return RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[],
        )


    def post_process_response(self, response: str, user_input: str) -> str:
        if "numéro" in user_input.lower() and "téléphone" in user_input.lower():
            phone_pattern = r'\b0\d{9}\b'
            phones = re.findall(phone_pattern, response)
            if phones:
                return f"Le numéro de téléphone trouvé est : {phones[0]}"
        return response

    def chat(self, user_input: str) -> str:
        self.memory.put(ChatMessage(role=MessageRole.USER, content=user_input))
        chat_history = self.memory.get()
        augmented_query = "Chat history:\n" + "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history]) + f"\nNew question: {user_input}"
        
        retriever_results = self.retriever.retrieve(augmented_query)
        Config.logger.info("\nRésultats du retriever:")
        for i, node in enumerate(retriever_results):
            Config.logger.info(f"Node {i+1}:")
            Config.logger.info(f"Source: {node.node.metadata.get('file_name', 'Unknown')}")
            Config.logger.info(f"Content: {node.node.text}")
            Config.logger.info(f"Score: {node.score}")
            Config.logger.info("-" * 50)

        response = self.chat_engine.query(augmented_query)
        processed_response = self.post_process_response(str(response), user_input)
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=processed_response))
        return processed_response

class RealTimeDataIntegrator:
    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager

    def integrate_new_data(self, new_data: List[str]):
        new_documents = [Document(text=data) for data in new_data]
        self.index_manager.update_index(new_documents)

class AdminInterface:
    def __init__(self, index_manager: IndexManager, data_integrator: RealTimeDataIntegrator):
        self.index_manager = index_manager
        self.data_integrator = data_integrator
        self.original_documents = {}
        self.load_original_documents()

    def load_original_documents(self):
        for filename in os.listdir(Config.DATA_DIR):
            file_path = os.path.join(Config.DATA_DIR, filename)
            try:
                docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
                if docs:
                    self.original_documents[filename] = docs[0]
                    Config.logger.info(f"Document loaded successfully: {filename}")
                else:
                    Config.logger.warning(f"No content found in file: {filename}")
            except Exception as e:
                Config.logger.error(f"Error loading file {filename}: {str(e)}")

    def add_new_document(self, file_path: str):
        file_path = file_path.strip('"')
        if not os.path.exists(file_path):
            Config.logger.error(f"File {file_path} does not exist.")
            return
        
        filename = os.path.basename(file_path)
        destination = os.path.join(Config.DATA_DIR, filename)
        shutil.copy2(file_path, destination)
        
        doc = SimpleDirectoryReader(input_files=[destination]).load_data()[0]
        self.original_documents[filename] = doc
        
        self.reindex_documents()
        Config.logger.info(f"Document {filename} added successfully.")

    def update_document(self, filename: str, new_content: str):
        if filename not in self.original_documents:
            Config.logger.error(f"Document {filename} not found.")
            return
        
        self.original_documents[filename].text = new_content
        with open(os.path.join(Config.DATA_DIR, filename), 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        self.reindex_documents()
        Config.logger.info(f"Document {filename} updated successfully.")

    def delete_document(self, filename: str):
        if filename not in self.original_documents:
            Config.logger.error(f"Document {filename} not found.")
            return
        
        del self.original_documents[filename]
        os.remove(os.path.join(Config.DATA_DIR, filename))
        
        self.reindex_documents()
        Config.logger.info(f"Document {filename} deleted successfully.")

    def reindex_documents(self):
        documents = list(self.original_documents.values())
        new_index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.index_manager.model_manager.service_context
        )
        self.index_manager.index = new_index
        self.index_manager.index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
        Config.logger.info(f"Index updated and persisted in {Config.PERSIST_DIR}")

    def list_documents(self):
        Config.logger.info("\nList of original documents:")
        for filename, doc in self.original_documents.items():
            Config.logger.info(f"Filename: {filename}")
            Config.logger.info(f"Content: {doc.text[:100]}...")
            Config.logger.info("-" * 50)
    def show_document_statistics(self):
        for filename, doc in self.original_documents.items():
            print(f"Filename: {filename}")
            print(f"Number of pages: {doc.metadata.get('num_pages', 'Unknown')}")
            print(f"Text length: {len(doc.text)} characters")
            print("-" * 50)

    def admin_menu(self):
        while True:
            print("\nAdmin Menu:")
            print("1. Add a new document")
            print("2. Update a document")
            print("3. Delete a document")
            print("4. List documents")
            print("5. Show document statistics")
            print("6. Return to chat")
            choice = input("Choose an option: ")

            if choice == '1':
                file_path = input("Enter the path of the file to add: ")
                self.add_new_document(file_path)
            elif choice == '2':
                self.list_documents()
                filename = input("Enter the name of the file to update: ")
                new_content = input("Enter the new content: ")
                self.update_document(filename, new_content)
            elif choice == '3':
                self.list_documents()
                filename = input("Enter the name of the file to delete: ")
                self.delete_document(filename)
            elif choice == '4':
                self.list_documents()
            elif choice == '5':
                self.show_document_statistics()
            elif choice == '6':
                break
            else:
                print("Invalid option. Please try again.")

class ChatbotInterface:
    def __init__(self, chatbot_engine: ChatbotEngine, data_integrator: RealTimeDataIntegrator, admin_interface: AdminInterface):
        self.chatbot_engine = chatbot_engine
        self.data_integrator = data_integrator
        self.admin_interface = admin_interface

    def main_loop(self):
        print("Bienvenue dans le chatbot. Tapez 'quit' pour quitter complètement le programme.")
        while True:
            self.chat_loop()
            restart = input("Voulez-vous commencer une nouvelle conversation? (oui/non): ").lower()
            if restart != 'oui':
                print("Merci d'avoir utilisé le chatbot. Au revoir!")
                break

    def chat_loop(self):
        print("\nNouvelle conversation commencée. Tapez 'exit' pour terminer cette conversation, 'update' pour intégrer de nouvelles données, ou 'admin' pour le menu administrateur.")
        while True:
            user_input = input("Utilisateur: ")
            if user_input.lower() == 'exit':
                print("Conversation terminée.")
                break
            elif user_input.lower() == 'update':
                self.update_data()
            elif user_input.lower() == 'admin':
                self.admin_interface.admin_menu()
            elif user_input.lower() == 'quit':
                print("Fermeture du programme.")
                exit()
            else:
                response = self.chatbot_engine.chat(user_input)
                print(f"AI: {response}")

    def update_data(self):
        new_data = input("Entrez les nouvelles données (séparées par des virgules): ").split(',')
        self.data_integrator.integrate_new_data(new_data)
        print("Données intégrées avec succès!")

def main():
    model_manager = ModelManager()
    index_manager = IndexManager(model_manager)
    chatbot_engine = ChatbotEngine(index_manager)  
    data_integrator = RealTimeDataIntegrator(index_manager)
    admin_interface = AdminInterface(index_manager, data_integrator)
    interface = ChatbotInterface(chatbot_engine, data_integrator, admin_interface)
    interface.main_loop()

if __name__ == "__main__":
    main()