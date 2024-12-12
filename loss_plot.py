import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Chemin du fichier
file_path = "checkpoints/KAIST_IRGAN_preprocess_v3/loss_log.txt"

# Lecture du fichier
with open(file_path, "r") as file:
    data = file.read()

# Extraction des données avec regex
pattern = r"\(epoch: (\d+), iters: (\d+),.*G_GAN: ([\d.]+) G_L1: ([\d.]+) G_Sobel: ([\d.]+) D_real: ([\d.]+) D_fake: ([\d.]+)"
matches = re.findall(pattern, data)

# Initialisation des variables pour calculer les moyennes par époque
metrics_per_epoch = defaultdict(lambda: defaultdict(list))

# Regroupement des métriques par époque
for match in matches:
    epoch = int(match[0])
    metrics_per_epoch[epoch]["G_GAN"].append(float(match[2]))
    metrics_per_epoch[epoch]["G_L1"].append(float(match[3]))
    metrics_per_epoch[epoch]["G_Sobel"].append(float(match[4]))
    metrics_per_epoch[epoch]["D_real"].append(float(match[5]))
    metrics_per_epoch[epoch]["D_fake"].append(float(match[6]))

# Calcul des moyennes par époque
average_metrics = {metric: [] for metric in ["G_GAN", "G_L1", "G_Sobel", "D_real", "D_fake"]}
epochs = sorted(metrics_per_epoch.keys())

for epoch in epochs:
    for metric in average_metrics:
        average_metrics[metric].append(sum(metrics_per_epoch[epoch][metric]) / len(metrics_per_epoch[epoch][metric]))

# Graphique 1 : G_GAN, D_real, D_fake
plt.figure(figsize=(10, 5))
plt.style.use('dark_background')

for key in ["G_GAN", "D_real", "D_fake"]:
    plt.plot(epochs, average_metrics[key], label=key)
plt.xlabel("Epochs")
plt.ylabel("Loss Values")
plt.title("Losses Per Epoch: G_GAN, D_real, D_fake")
plt.legend()
plt.grid()
plt.savefig('avg_loss_gan_real_fake.png')

# Graphique 2 : G_L1, G_Sobel
plt.figure(figsize=(10, 5))
plt.style.use('dark_background')

for key in ["G_L1", "G_Sobel"]:
    plt.plot(epochs, average_metrics[key], label=key)
plt.xlabel("Epochs")
plt.ylabel("Loss Values")
plt.title("Losses Per Epoch: G_L1, G_Sobel")
plt.legend()
plt.grid()
plt.savefig('avg_loss_l1_sobel.png')
