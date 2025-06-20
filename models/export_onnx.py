import torch
from driver_model import DriverMonitorModel

model = DriverMonitorModel()
ckpt = torch.load('models/epoch_24_ckpt.pth.tar', map_location='cpu')
model.load_state_dict(ckpt['state_dict'])
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "models/driver_model.onnx",
                  input_names=['input'], output_names=['output'],
                  opset_version=11)
print("✅ ONNX export complete.")
