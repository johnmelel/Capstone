# Test import
try:
    from magic_pdf.pipe.UNIPipe import UNIPipe
    from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
    print("✅ MinerU installed successfully!")
except ImportError as e:
    print(f"❌ Installation issue: {e}")

    