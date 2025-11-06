
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_status(check_name, success, details=""):
    """Print status of a check."""
    status = "✅" if success else "❌"
    print(f"\n{status} {check_name}")
    if details:
        print(f"   {details}")

# =============================================================================
# 1. PYTHON VERSION CHECK
# =============================================================================
print_header("PYTHON VERSION CHECK")

python_version = sys.version_info
version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
print(f"Python version: {version_str}")

if 3.10 <= python_version.major + python_version.minor/10 <= 3.13:
    print_status("Python version", True, f"Version {version_str} is supported")
else:
    print_status("Python version", False, 
                 f"Version {version_str} may not be fully supported. Recommended: 3.10-3.13")

# =============================================================================
# 2. CORE DEPENDENCIES CHECK
# =============================================================================
print_header("CORE DEPENDENCIES CHECK")

# Check numpy
try:
    import numpy as np
    numpy_version = np.__version__
    if np.__version__.startswith('2.'):
        print_status("NumPy", False, 
                     f"Version {numpy_version} is too new. MinerU needs <2.0.0")
    else:
        print_status("NumPy", True, f"Version {numpy_version}")
except ImportError:
    print_status("NumPy", False, "Not installed")

# Check Pillow
try:
    from PIL import Image
    import PIL
    print_status("Pillow (PIL)", True, f"Version {PIL.__version__}")
except ImportError:
    print_status("Pillow (PIL)", False, "Not installed")

# Check dotenv
try:
    import dotenv
    print_status("python-dotenv", True, "Installed")
except ImportError:
    print_status("python-dotenv", False, "Not installed")

# Check PyYAML
try:
    import yaml
    print_status("PyYAML", True, "Installed")
except ImportError:
    print_status("PyYAML", False, "Not installed")

# =============================================================================
# 3. MINERU INSTALLATION CHECK
# =============================================================================
print_header("MINERU INSTALLATION CHECK")

# Check if mineru package is installed
try:
    import mineru
    print_status("MinerU package", True, "Package installed")
    
    # Check mineru version using CLI
    try:
        result = subprocess.run(['mineru', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_status("MinerU CLI", True, f"Version: {version}")
        else:
            print_status("MinerU CLI", False, "CLI available but version check failed")
    except FileNotFoundError:
        print_status("MinerU CLI", False, "Command 'mineru' not found in PATH")
    except Exception as e:
        print_status("MinerU CLI", False, f"Error: {e}")
    
    # Check if we can import specific components
    try:
        from mineru.data.data_reader_writer import FileBasedDataWriter
        print_status("MinerU components", True, "Can import internal modules")
    except ImportError as e:
        print_status("MinerU components", False, f"Cannot import: {e}")
        
except ImportError:
    print_status("MinerU package", False, "Not installed. Run: pip install 'mineru[core]'")

# =============================================================================
# 4. VERTEX AI CHECK
# =============================================================================
print_header("VERTEX AI CHECK")

try:
    from google.cloud import aiplatform
    print_status("google-cloud-aiplatform", True, "Installed")
except ImportError:
    print_status("google-cloud-aiplatform", False, "Not installed")

try:
    import vertexai
    print_status("vertexai", True, "Installed")
except ImportError:
    print_status("vertexai", False, "Not installed")

# =============================================================================
# 5. MILVUS CHECK
# =============================================================================
print_header("MILVUS CHECK")

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
    print_status("pymilvus", True, "Installed")
except ImportError:
    print_status("pymilvus", False, "Not installed")

# =============================================================================
# 6. MODEL FILES CHECK
# =============================================================================
print_header("MODEL FILES CHECK")

# Check if models directory exists
mineru_models_path = Path.home() / ".mineru" / "models"
if mineru_models_path.exists():
    model_files = list(mineru_models_path.glob("**/*"))
    model_count = len([f for f in model_files if f.is_file()])
    print_status("Model files", True, 
                 f"Found {model_count} model files in {mineru_models_path}")
    print(f"\n   To download models, run: mineru download-models")
else:
    print_status("Model files", False, 
                 f"Models directory not found: {mineru_models_path}")
    print(f"\n   Download models with: mineru download-models")

# =============================================================================
# 7. POTENTIAL ISSUES CHECK
# =============================================================================
print_header("CHECKING FOR COMMON ISSUES")

# Check for old magic-pdf installation
try:
    import magic_pdf
    print_status("Old magic-pdf package", False, 
                 "Found old 'magic-pdf' package. Uninstall it: pip uninstall magic-pdf -y")
except ImportError:
    print_status("No old magic-pdf", True, "Old package not found (good!)")

# Check for detectron2
try:
    import detectron2
    print_status("detectron2", False, 
                 "Found detectron2 (not needed for MinerU 2.0+). Uninstall: pip uninstall detectron2 -y")
except ImportError:
    print_status("No detectron2", True, "detectron2 not found (good!)")

# =============================================================================
# 8. SUMMARY
# =============================================================================
print_header("INSTALLATION SUMMARY")

print("""
Next steps:
1. If all checks pass, download models:
   mineru download-models

2. Test MinerU on a PDF:
   mineru -p your_file.pdf -o output_dir

3. If you see errors about numpy version:
   pip uninstall numpy -y
   pip install 'numpy>=1.21.6,<2.0.0'
   pip install 'mineru[core]' --force-reinstall

4. For API usage in Python:
   from mineru.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
   # MinerU 2.0+ is primarily a CLI tool, not a Python library

5. Check the docs: https://opendatalab.github.io/MinerU/
""")

print("="*70)
print("  Installation check complete!")
print("="*70 + "\n")