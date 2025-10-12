from protocols import A2AMessage, MCPRequest, MCPResponse
from vector_store import VectorStoreClient
import config

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.mailbox: list[A2AMessage] = []

    def send(self, msg: A2AMessage, manager):
        manager.deliver(msg)

    def receive(self, msg: A2AMessage):
        self.mailbox.append(msg)

    def step(self, manager):
        raise NotImplementedError

class TextRetrieverAgent(BaseAgent):
    def __init__(self, store: VectorStoreClient):
        super().__init__('text_retriever')
        self.store = store

    def step(self, manager):
        if not self.mailbox: return
        msg = self.mailbox.pop(0)
        query_vec = msg.content['query_vector']
        hits = self.store.query(query_vec)
        reply = A2AMessage(self.name, msg.sender, {'text_hits': hits})
        self.send(reply, manager)

class ImageRetrieverAgent(BaseAgent):
    def __init__(self, store: VectorStoreClient):
        super().__init__('image_retriever')
        self.store = store

    def step(self, manager):
        if not self.mailbox: return
        msg = self.mailbox.pop(0)
        query_vec = msg.content['query_vector']
        hits = self.store.query(query_vec)
        reply = A2AMessage(self.name, msg.sender, {'image_hits': hits})
        self.send(reply, manager)

class FusionAgent(BaseAgent):
    def step(self, manager):
        if not self.mailbox: return
        msg = self.mailbox.pop(0)
        text = msg.content.get('text_hits', [])
        images = msg.content.get('image_hits', [])

        # simple fusion bc im just concatenate lists
        fused = text + images
        reply = A2AMessage(self.name, msg.sender, {'fused_hits': fused})
        self.send(reply, manager)

class PromptGeneratorAgent(BaseAgent):
    def step(self, manager):
        if not self.mailbox: return
        msg = self.mailbox.pop(0)
        base = msg.content['user_query']
        generated = f"Retrieve relevant text and images for: {base}"
        reply = A2AMessage(self.name, 'text_retriever', {'query_vector': msg.content['text_vector']})

        # also send to image retriever
        manager.deliver(A2AMessage(self.name, 'image_retriever', {'query_vector': msg.content['image_vector']}))
        self.send(reply, manager)

class ReasonerAgent(BaseAgent):
    def step(self, manager):
        if not self.mailbox: return
        msg = self.mailbox.pop(0)
        docs = msg.content['fused_hits']
        combined_text = ''.join(d['metadata']['text'] for d in docs if 'text' in d['metadata'])
        combined_summary = combined_text[:500]
        reply = A2AMessage(self.name, 'user', {'response': combined_summary})
        self.send(reply, manager)