# from models.experimental import attempt_load
import torch

# weights_path = "./algorithm/coco.weights"

# model = attempt_load(weights_path, device='cpu')

# torch.save(model, 'model.pt')

model = torch.load('model.pt')

state_dict = model.state_dict()

for key, value in state_dict.items():
    print(key, type(value), value.size())