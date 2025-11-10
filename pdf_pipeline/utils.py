import re

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename by removing or replacing invalid characters.
    """
    # Remove leading/trailing whitespace
    filename = filename.strip()
    # Replace invalid characters with an underscore
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return filename
