import os
import kagglehub
import matplotlib.pyplot as plt
from PIL import Image

# Télécharger et récupérer le chemin du dataset
path = kagglehub.dataset_download("alessiocorrado99/animals10")
print("Path to dataset files:", path)

# Le dossier contenant réellement les images
image_root = os.path.join(path, "raw-img")

# Vérifier si le dossier existe
if not os.path.exists(image_root):
    raise ValueError(f"Le dossier '{image_root}' n'existe pas.")

# Lister les catégories (ex: 'cane', 'cat', 'dog', etc.)
categories = os.listdir(image_root)
print("Catégories disponibles :", categories)

# Sélectionner quelques images et les afficher
num_images = 5
plt.figure(figsize=(10, 5))

for i, category in enumerate(categories[:num_images]):  
    category_path = os.path.join(image_root, category)
    
    if os.path.isdir(category_path):  # Vérifie si c'est un dossier
        image_files = os.listdir(category_path)
        
        if image_files:  # Vérifie qu'il y a des images
            img_path = os.path.join(category_path, image_files[0])  # Prendre une image
            img = Image.open(img_path)

            # Affichage
            plt.subplot(1, num_images, i + 1)
            plt.imshow(img)
            plt.title(category)
            plt.axis("off")

plt.show()
