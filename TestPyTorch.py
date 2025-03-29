import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example: Move a torchvision model to the correct device
model = torchvision.models.resnet18(pretrained=True)
model = model.to(device)

print(torch.__version__)
print(torchvision.__version__)