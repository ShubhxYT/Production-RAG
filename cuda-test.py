import torch

print("PyTorch version:", torch.__version__)

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    print("GPU is available!")
    print("Device name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
    
    device = torch.device("cuda")
else:
    print("GPU not available, using CPU.")
    device = torch.device("cpu")

print("Selected device:", device)

# Simple tensor test
x = torch.rand(3, 3).to(device)
y = torch.rand(3, 3).to(device)

z = x + y
print("Tensor device:", z.device)
print(z)