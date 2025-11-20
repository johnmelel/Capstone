import logging
import google.generativeai as genai
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
            self.model = None
            return
            
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_ANNOTATION_MODEL)
        logger.info(f"GeminiAnnotator initialized with model: {Config.GEMINI_ANNOTATION_MODEL}")
        
    def annotate_image(self, image_bytes: bytes, prompt: str) -> Optional[str]:
        """
        Send image to Gemini for annotation.
        
        Args:
            image_bytes: Raw bytes of the image
            prompt: Instruction for Gemini
            
        Returns:
            Generated text or None if failed
        """
        if not self.model:
            return None
            
        try:
            # Create image part
            image_part = {
                "mime_type": "image/jpeg", # Assuming JPEG/PNG, Gemini handles common formats
                "data": image_bytes
            }
            
            response = self.model.generate_content([prompt, image_part])
            
            if response.text:
                return response.text.strip()
            return None
            
        except Exception as e:
            logger.error(f"Error annotating image with Gemini: {e}")
            return None
