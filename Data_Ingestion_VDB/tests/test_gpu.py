"""
Quick test to check if MinerU can use your M3 GPU.

This script checks:
1. If PyTorch is installed
2. If Metal Performance Shaders (MPS) is available on your M3
3. If CUDA is available (it won't be on Mac)
4. What device MinerU will use
"""

import sys

print("="*70)
print("GPU AVAILABILITY CHECK")
print("="*70)

# Check PyTorch
print("\n[1] Checking PyTorch installation...")
try:
    import torch
    print(f"   ✓ PyTorch installed: version {torch.__version__}")
except ImportError:
    print("   ✗ PyTorch not installed")
    sys.exit(1)

# Check MPS (Metal Performance Shaders - M3's GPU interface)
print("\n[2] Checking M3 GPU (MPS) availability...")
if torch.backends.mps.is_available():
    print("   ✓ MPS (M3 GPU) is AVAILABLE!")
    print("   ✓ MinerU can use your M3 GPU for acceleration")
    mps_built = torch.backends.mps.is_built()
    print(f"   ✓ MPS backend built: {mps_built}")
else:
    print("   ✗ MPS not available (will use CPU)")

# Check CUDA (NVIDIA GPUs - won't be available on Mac)
print("\n[3] Checking CUDA (NVIDIA GPU)...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("   ✗ CUDA not available (expected on Mac)")

# Determine default device
print("\n[4] Default device for MinerU...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"   → Device: MPS (M3 GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"   → Device: CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print(f"   → Device: CPU")

# Test tensor creation on device
print("\n[5] Testing tensor creation on device...")
try:
    test_tensor = torch.randn(100, 100).to(device)
    print(f"   ✓ Successfully created tensor on {device}")
    print(f"   ✓ Tensor shape: {test_tensor.shape}")
    print(f"   ✓ Tensor device: {test_tensor.device}")
except Exception as e:
    print(f"   ✗ Failed to create tensor: {e}")

# MinerU specific check
print("\n[6] Checking MinerU installation...")
try:
    from magic_pdf.pipe.UNIPipe import UNIPipe
    print("   ✓ MinerU (magic-pdf) is installed")
    print("   ✓ MinerU will use the device shown above")
except ImportError:
    print("   ✗ MinerU not installed")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if torch.backends.mps.is_available():
    print("✓ Your M3 GPU is available and will be used!")
    print("✓ MinerU will run faster with GPU acceleration")
else:
    print("✗ GPU not available, will use CPU")
    print("  (This is fine, just slower)")
print("="*70)