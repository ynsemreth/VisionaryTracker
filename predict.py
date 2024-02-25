# import matplotlib.pyplot as plt
# from algorithm.object_detector import YOLOv7

# def plot_weights_histogram(model, layer_index=0):
#     layer_weights = list(model.parameters())[layer_index].detach().cpu().numpy()
    
#     # Ağırlıkların histogramını çiz
#     plt.hist(layer_weights.ravel(), bins=100)
#     plt.title(f'Layer {layer_index} Weight Distribution')
#     plt.xlabel('Weight Value')
#     plt.ylabel('Frequency')
#     plt.show()

# model = YOLOv7()
# model.load('coco.weights', 'coco.yaml', device='cpu')

# plot_weights_histogram(model.model, 0)


from tensorflow.keras.models import load_model

# Modeli Yükle
model = load_model('modelinizin_yolu.h5')

# Modelin Katmanlarını Listele
model.summary()

# Belirli bir katmanın ağırlıklarına eriş
for layer in model.layers:
    weights = layer.get_weights()  # Bu, bir liste döndürür
    print(f"{layer.name} katmanının ağırlıkları:")
    for weight in weights:
        print(weight)  # Bu, numpy dizisidir
