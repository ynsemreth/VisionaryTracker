import matplotlib.pyplot as plt

layers = [
    "model.0.conv",
    "model.1.conv",
    "model.2.conv",
    "model.3.conv",
    "model.4.conv",
    "model.5.conv",
    "model.7.conv",
    "model.9.conv",
    "model.11.conv",
    "model.14.conv",
    "model.18.conv",
    "model.21.conv",
    "model.25.conv",
    "model.28.conv",
    "model.35.conv",
    "model.38.conv",
    "model.42.conv",
    "model.44.conv",
    "model.47.conv",
    "model.50.conv",
    "model.52.conv",
    "model.54.conv",
    "model.57.conv",
    "model.60.conv",
    "model.62.conv",
    "model.65.conv",
    "model.68.conv",
    "model.70.conv",
    "model.73.conv",
    "model.75.conv",
    "model.76.conv",
    "model.77.m.0",
]

weights_size = [
    32, 64, 32, 32, 32, 32,
    64, 64, 64, 128, 128, 256,
    256, 512, 256, 128, 64,
    64, 128, 64, 32, 32, 64,
    64, 64, 128, 128, 128, 256,
    256, 512, 255,
]

plt.figure(figsize=(10, 8))
plt.barh(layers, weights_size, color='skyblue')
plt.xlabel('Parametre Sayısı (ağırlıkların boyutu)')
plt.ylabel('Katmanlar')
plt.title('Model Katmanlarının Ağırlık Boyutları')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
