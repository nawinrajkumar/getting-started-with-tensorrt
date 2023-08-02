# import libraries
import torch
import torchvision.models as models
import torch.onnx

# import the model
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)

# Setting up Batch size
BATCH_SIZE = 32
dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)
torch.onnx.export(resnext50_32x4d, dummy_input, "resnet50_onnx_model.onnx", verbose=False)
