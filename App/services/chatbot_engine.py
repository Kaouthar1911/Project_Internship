from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from .index_manager import IndexManager

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

        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[],
        )

    def chat(self, user_input: str) -> str:
        self.memory.put(ChatMessage(role=MessageRole.USER, content=user_input))
        chat_history = self.memory.get()
        augmented_query = f"Chat history:\n"
        for msg in chat_history:
            augmented_query += f"{msg.role}: {msg.content}\n"
        augmented_query += f"\nNew question: {user_input}"
        
        response = self.chat_engine.query(augmented_query)
        
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=str(response)))
        return str(response.response)

    def reset(self):
        self.memory.clear()