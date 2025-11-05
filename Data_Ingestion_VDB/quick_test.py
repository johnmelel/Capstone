# Create this file: quick_test.py
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pathlib import Path

pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True

converter = DocumentConverter()  # No pipeline_options argument
result = converter.convert("test_dataset/1-s2.0-S0720048X23000712-main.pdf")

pics = list(result.document.pictures)
print(f"Images found: {len(pics)}")

if len(pics) > 0:
    print("✅ YES - Images ARE being extracted!")
else:
    print("❌ NO - No images found in this PDF")