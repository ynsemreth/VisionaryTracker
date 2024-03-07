import numpy as np
import matplotlib.pyplot as plt
from models.experimental import attempt_load

weights_path = "./algorithm/coco.weights"

model = attempt_load(weights_path, device='cpu')
model.eval()

first_conv_weights = next(model.named_parameters())[1].data.cpu().numpy()

num_filters = first_conv_weights.shape[0]

fig, axes = plt.subplots(1, num_filters, figsize=(num_filters*2, 2))
for i, ax in enumerate(axes):
    weight = first_conv_weights[i]
    weight_min, weight_max = weight.min(), weight.max()
    weight = (weight - weight_min) / (weight_max - weight_min)
  
    ax.imshow(weight.transpose((1, 2, 0)), interpolation='nearest')
    ax.axis('off')
plt.show()