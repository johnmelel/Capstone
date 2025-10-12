from transformers import AutoTokenizer, AutoModel
import torch
import config
from torchvision import transforms
from PIL import Image
import clip

class TextEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_TEXT)
        self.model = AutoModel.from_pretrained(config.EMBEDDING_MODEL_TEXT)

    def embed(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().tolist()
        return embeddings

class ImageEmbedder:
    def __init__(self):
        self.preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
        ])
        self.model, _ = clip.load(config.EMBEDDING_MODEL_IMAGE)

    def embed(self, pil_images: list[Image.Image]) -> list[list[float]]:
        inputs = torch.stack([self.preprocess(img) for img in pil_images])
        with torch.no_grad():
            embeds = self.model.encode_image(inputs)
        return embeds.detach().tolist()