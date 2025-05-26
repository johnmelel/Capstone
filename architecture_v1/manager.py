from agents import (
    TextRetrieverAgent, ImageRetrieverAgent,
    FusionAgent, PromptGeneratorAgent, ReasonerAgent
)
from vector_store import VectorStoreClient
from embedder import TextEmbedder, ImageEmbedder
from rl_trainer import RLTrainer
from protocols import A2AMessage
import config

class AgentManager:
    def __init__(self):
        self.text_store = VectorStoreClient(config.VECTOR_DB_COLLECTION_TEXT)
        self.image_store = VectorStoreClient(config.VECTOR_DB_COLLECTION_IMAGE)
        self.text_retr = TextRetrieverAgent(self.text_store)
        self.img_retr = ImageRetrieverAgent(self.image_store)
        self.fusion = FusionAgent('fusion')
        self.prompt_gen = PromptGeneratorAgent('prompt_gen')
        self.reasoner = ReasonerAgent('reasoner')
        self.embed_text = TextEmbedder()
        self.embed_img = ImageEmbedder()
        self.rl = RLTrainer()
        self.agents = {
            'text_retriever': self.text_retr,
            'image_retriever': self.img_retr,
            'fusion': self.fusion,
            'prompt_gen': self.prompt_gen,
            'reasoner': self.reasoner
        }

    def deliver(self, msg: A2AMessage):
        agent = self.agents.get(msg.receiver)
        if agent:
            agent.receive(msg)

    def run(self, user_query: str, images: list):
        text_vec = self.embed_text.embed([user_query])[0]
        img_vecs = self.embed_img.embed(images)
        # send initial query to prompt generator
        init = A2AMessage('user', 'prompt_gen', {
            'user_query': user_query,
            'text_vector': text_vec,
            'image_vector': img_vecs[0]
        })
        self.deliver(init)
        # run steps in DAG order
        for _ in range(3):
            for agent in self.agents.values():
                agent.step(self)