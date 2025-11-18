# PDF Splitter Service

Standalone service to split large PDFs in Google Cloud Storage into smaller parts (~25MB each).

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Uses existing .env from parent directory
# Make sure your main .env has:
#   GCS_BUCKET_NAME=your-bucket
#   GOOGLE_SERVICE_ACCOUNT_JSON=path/to/service-account.json
#   GCS_BUCKET_PREFIX=optional/folder/path (optional)

# 4. Run
python pdf_splitter.py --dry-run  # Preview first
python pdf_splitter.py            # Actually split
```

## Usage

```bash
# Basic usage
python pdf_splitter.py

# Custom target size (20MB parts)
python pdf_splitter.py --target-size 20

# Process specific folder
python pdf_splitter.py --prefix "medical-journals/"

# All options
python pdf_splitter.py --bucket my-bucket --prefix docs/ --target-size 20 --min-size 40 --dry-run
```

## Options

- `--service-account PATH` - GCS service account JSON
- `--bucket NAME` - GCS bucket name
- `--prefix PATH` - Folder prefix to scan
- `--target-size MB` - Target size per part (default: 25)
- `--min-size MB` - Min size to split (default: 30)
- `--dry-run` - Preview only

## Output

Original: `documents/paper.pdf` (80MB)
↓
Split: 
- `documents/paper_split/paper_part_1_of_3.pdf` (27MB)
- `documents/paper_split/paper_part_2_of_3.pdf` (27MB)
- `documents/paper_split/paper_part_3_of_3.pdf` (26MB)
