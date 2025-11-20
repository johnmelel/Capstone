import logging
from google import genai
from google.genai import types
from typing import Optional
from .config import Config

logger = logging.getLogger(__name__)

class GeminiAnnotator:
    """
    Uses Gemini API to generate captions/descriptions for images and tables.
    """
    
    def __init__(self):
        if not Config.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not found. GeminiAnnotator will be disabled.")
            self.client = None
            return
            
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.model_name = Config.GEMINI_ANNOTATION_MODEL
        logger.info(f"GeminiAnnotator initialized with model: {self.model_name}")
        
    def annotate_image(self, image_bytes: bytes, prompt: str) -> Optional[str]:
        """
        Send image to Gemini for annotation.
        
        Args:
            image_bytes: Raw bytes of the image
            prompt: Instruction for Gemini
            
        Returns:
            Generated text or None if failed
        """
        if not self.client:
            return None
            
        try:
            # Create image part using types
            # We assume JPEG for simplicity, but Gemini handles most formats
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image_part]
            )
            
            if response.text:
                return response.text.strip()
            return None
            
        except Exception as e:
            logger.error(f"Error annotating image with Gemini: {e}")
            return None
