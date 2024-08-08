import os
from llama_index.core.llms import ChatMessage, MessageRole
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
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
import os
from llama_index.core.llms import ChatMessage, MessageRole
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
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

class Config:
    PERSIST_DIR = "./st"
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama3-70b-8192"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 20
    DATA_DIR = "./data"

class ModelManager:
    def __init__(self):
        self.embed_model = HuggingFaceEmbedding(model_name=Config.EMBED_MODEL_NAME)
        self.llm = Groq(model=Config.LLM_MODEL, api_key=os.getenv('GROQ_API_KEY'))
        self.text_splitter = SentenceSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        Settings.text_splitter = self.text_splitter
        self.service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model)
        self.service_context=self.service_context



class IndexManager:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.documents = self.load_documents()
        self.display_documents()
        self.index = self.load_index_from_storage()

    def load_documents(self):
        return SimpleDirectoryReader(input_dir=Config.DATA_DIR).load_data()

    def display_documents(self):
        print("\nDocuments chargés:")
        for i, doc in enumerate(self.documents, 1):
            print(f"\nDocument {i}:")
            print(f"Nom du fichier: {doc.metadata.get('file_name', 'Non spécifié')}")
            print(f"Type de fichier: {doc.metadata.get('file_type', 'Non spécifié')}")
            print(f"num de page: {doc.metadata.get('page_label', 'Non spécifié')}")
            print(f"Contenu (premiers 500 caractères): {doc.text[:500]}...")
            print("-" * 50)

    def load_index_from_storage(self):
        if os.path.exists(Config.PERSIST_DIR):
            print("Chargement de l'index existant depuis le stockage...")
            storage_context = StorageContext.from_defaults(persist_dir=Config.PERSIST_DIR)
            return load_index_from_storage(
                storage_context, 
                service_context=self.model_manager.service_context
            )
        else:
            print("Aucun index existant trouvé. Création d'un nouvel index...")
            return self.create_new_index()

    def create_new_index(self):
        index = VectorStoreIndex.from_documents(
            self.documents, 
            embed_model=self.model_manager.embed_model, 
            transformations=[self.model_manager.text_splitter],
            # service_context=self.model_manager.service_context
        )
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
        return index

    def update_index(self, new_documents: List[Document]):
        self.documents.extend(new_documents)
        self.display_documents()  # Afficher les documents mis à jour
        for doc in new_documents:
            self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=Config.PERSIST_DIR)

class ChatbotEngine:
    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        self.chat_engine = self.initialize_chat_engine()

    def initialize_chat_engine(self):
        retriever = VectorIndexRetriever(
            index=self.index_manager.index,
            similarity_top_k=20,
        )

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
        prompt_template = PromptTemplate(template=template)

        response_synthesizer = get_response_synthesizer(
            response_mode="simple_summarize",
            service_context=self.index_manager.model_manager.service_context,
            text_qa_template=prompt_template,
            use_async=False,
            streaming=False,
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[],
        )
        

        return query_engine

    def chat(self, user_input: str) -> str:
        self.memory.put(ChatMessage(role=MessageRole.USER, content=user_input))
        chat_history = self.memory.get()
        augmented_query = f"Chat history:\n"
        for msg in chat_history:
            augmented_query += f"{msg.role}: {msg.content}\n"
        augmented_query += f"\nNew question: {user_input}"
        
        # Afficher les résultats du retriever
        self.print_retrieved_context(augmented_query)
        
        # Use query() instead of chat()
        response = self.chat_engine.query(augmented_query)
        
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=str(response)))
        return str(response.response)
    
    def print_retrieved_context(self, question):
        retrieved_nodes = self.chat_engine.retriever.retrieve(question)
        print("\nContexte récupéré par le retriever:")
        for i, node in enumerate(retrieved_nodes):
            print(f"Node {i + 1}:")
            print(node.get_content())
            print("-" * 30)

    # def reset(self):
    #     self.memory.clear()

class RealTimeDataIntegrator:
    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager

    def integrate_new_data(self, new_data: List[str]):
        new_documents = [Document(text=data) for data in new_data]
        self.index_manager.update_index(new_documents)

class ChatbotInterface:
    def __init__(self, chatbot_engine: ChatbotEngine, data_integrator: RealTimeDataIntegrator):
        self.chatbot_engine = chatbot_engine
        self.data_integrator = data_integrator

    def chat_loop(self):
        print("Conversation commencée. Tapez 'exit' pour terminer ou 'update' pour intégrer de nouvelles données.")
        while True:
            user_input = input("Utilisateur: ")
            if user_input.lower() == 'exit':
                self.chat_engine.reset()
                break
            elif user_input.lower() == 'update':
                self.update_data()
            else:
                response = self.chatbot_engine.chat(user_input)
                print(f"AI: {response}")

    def update_data(self):
        new_data = input("Entrez les nouvelles données (séparées par des virgules): ").split(',')
        self.data_integrator.integrate_new_data(new_data)
        print("Données intégrées avec succès!")



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
                    for i, doc in enumerate(docs):
                        doc_key = f"{filename}_part_{i+1}" if len(docs) > 1 else filename
                        self.original_documents[doc_key] = doc
                    print(f"Document loaded successfully: {filename} with {len(docs)} part(s)")
                else:
                    print(f"No content found in file: {filename}")
            except Exception as e:
                print(f"Error loading file {filename}: {str(e)}")

    def select_multiple_files_gui(self) -> List[str]:
            
            root = tk.Tk()
            root.withdraw()  # Cacher la fenêtre principale
            file_paths = filedialog.askopenfilenames(
                title="Sélectionnez les fichiers à télécharger",
                filetypes=[("Tous les fichiers", "*.*")]
            )
            return list(file_paths)

    def add_new_document(self, file_path: str):
        file_path = file_path.strip('"')
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return
        
        filename = os.path.basename(file_path)
        destination = os.path.join(Config.DATA_DIR, filename)
        shutil.copy2(file_path, destination)
        
        docs = SimpleDirectoryReader(input_files=[destination]).load_data()
        if docs:
            # Ajouter tous les documents chargés
            for i, doc in enumerate(docs):
                doc_key = f"{filename}_part_{i+1}" if len(docs) > 1 else filename
                self.original_documents[doc_key] = doc
            
            self.reindex_documents()
            print(f"Document {filename} added successfully with {len(docs)} part(s).")
        else:
            print(f"No content found in file {filename}")

    def update_document(self, filename: str, new_content: str):
        if filename not in self.original_documents:
            print(f"Document {filename} not found.")
            return
        
        self.original_documents[filename].text = new_content
        with open(os.path.join(Config.DATA_DIR, filename), 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        self.reindex_documents()
        print(f"Document {filename} updated successfully.")

    def delete_document(self, filename: str):
        if filename not in self.original_documents:
            print(f"Document {filename} not found.")
            return
        
        del self.original_documents[filename]
        os.remove(os.path.join(Config.DATA_DIR, filename))
        
        self.reindex_documents()
        print(f"Document {filename} deleted successfully.")


    def process_single_file(self, file_path: str, data_dir: str):
        filename = os.path.basename(file_path)
        destination = os.path.join(data_dir, filename)
        
        try:
            shutil.copy2(file_path, destination)
            docs = SimpleDirectoryReader(input_files=[destination]).load_data()
            if docs:
                return filename, docs, None
            else:
                return filename, None, "No content found in file"
        except Exception as e:
            if os.path.exists(destination):
                os.remove(destination)
            return filename, None, str(e)

    def upload_multiple_files(self, file_paths: List[str]):
        added_files = []
        errors = []

        with ProcessPoolExecutor() as executor:
            process_file = partial(self.process_single_file, data_dir=Config.DATA_DIR)
            future_to_file = {executor.submit(process_file, file_path): file_path for file_path in file_paths}
            
            for future in as_completed(future_to_file):
                filename, docs, error = future.result()
                if error:
                    errors.append(f"Erreur lors du chargement du fichier {filename}: {error}")
                else:
                    for i, doc in enumerate(docs):
                        doc_key = f"{filename}_part_{i+1}" if len(docs) > 1 else filename
                        self.original_documents[doc_key] = doc
                    added_files.append(filename)
                    print(f"Fichier {filename} téléchargé avec succès avec {len(docs)} partie(s).")

        if added_files:
            self.reindex_documents()
            print(f"Ajout réussi de {len(added_files)} documents : {', '.join(added_files)}")
        else:
            print("Aucun document n'a été ajouté.")

        if errors:
            print("\nErreurs rencontrées:")
            for error in errors:
                print(error)

    def reindex_documents(self):
        documents = list(self.original_documents.values())
        new_index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.index_manager.model_manager.service_context
        )
        self.index_manager.index = new_index
        self.index_manager.index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
        print(f"Index mis à jour et persisté dans {Config.PERSIST_DIR}")
        print(f"Nombre total de documents indexés : {len(documents)}")

    def list_documents(self):
        print("\nList of original documents:")
        for filename, doc in self.original_documents.items():
            print(f"Filename: {filename}")
            print(f"Content: {doc.text[:100]}...")  # Display first 100 characters
            print("-" * 50)

    def admin_menu(self):
        while True:
            print("\nMenu Administrateur:")
            print("1. Ajouter un nouveau document")
            print("2. Télécharger plusieurs documents")
            print("3. Mettre à jour un document")
            print("4. Supprimer un document")
            print("5. Lister les documents")
            print("6. Retourner au chat")
            choice = input("Choisissez une option : ")

            if choice == '1':
                file_path = input("Entrez le chemin du fichier à ajouter : ")
                self.add_new_document(file_path)
            elif choice == '2':
                file_paths = self.select_multiple_files_gui()
                if file_paths:
                    self.upload_multiple_files(file_paths)
                else:
                    print("Aucun fichier sélectionné.")
            elif choice == '3':
                self.list_documents()
                filename = input("Entrez le nom du fichier à mettre à jour : ")
                new_content = input("Entrez le nouveau contenu : ")
                self.update_document(filename, new_content)
            elif choice == '4':
                self.list_documents()
                filename = input("Entrez le nom du fichier à supprimer : ")
                self.delete_document(filename)
            elif choice == '5':
                self.list_documents()
            elif choice == '6':
                break
            else:
                print("Option invalide. Veuillez réessayer.")

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
                self.chatbot_engine.reset()
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