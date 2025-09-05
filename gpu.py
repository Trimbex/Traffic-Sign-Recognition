# Quick CUDA check
import torch
import tensorflow as tf

print("=== PyTorch CUDA Check ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

print("\n=== TensorFlow GPU Check ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# Check for GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow GPU devices: {gpus}")

if gpus:
    print("GPU is detected by TensorFlow!")
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu}")
        
    # Test GPU with a simple operation
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
        print("✅ GPU computation test successful!")
        print(f"Test result: {c}")
    except Exception as e:
        print(f"❌ GPU computation test failed: {e}")
else:
    print("❌ No GPU detected by TensorFlow")
    
    # Check if this is a CUDA/cuDNN issue
    print("\n=== Diagnostic Information ===")
    print("Possible causes:")
    print("1. Wrong TensorFlow version (need tensorflow-gpu or tensorflow>=2.0 with GPU support)")
    print("2. Missing or incompatible CUDA drivers")
    print("3. Missing or incompatible cuDNN libraries")
    print("4. Environment variables not set correctly")
    
    # Check CUDA availability from system
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("\n✅ nvidia-smi works - NVIDIA drivers are installed")
            print("This suggests the issue is with TensorFlow-CUDA integration")
        else:
            print("\n❌ nvidia-smi failed - possible driver issue")
    except Exception as e:
        print(f"\n❌ Could not run nvidia-smi: {e}")
    
    print(f"\nYour PyTorch can see CUDA, so the GPU and drivers are working.")
    print(f"The issue is likely with TensorFlow's CUDA integration.")