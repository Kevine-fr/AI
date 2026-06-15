from sklearn.tree import DecisionTreeClassifier

# Données d'entraînement (taille, poids)
# 1 : Pomme, 0 : Orange
features = [[3, 100], [3.5, 120], [4, 150], [2.5, 80], [3, 85]]
labels = [1, 1, 1, 0, 0]  # 1 = Pomme, 0 = Orange

# Création du modèle
model = DecisionTreeClassifier()

# Entraînement du modèle
model.fit(features, labels)

# Données de test
test_data = [[3.3, 110], [2.8, 90]]

# Prédiction
predictions = model.predict(test_data)

# Affichage des résultats
for fruit, prediction in zip(test_data, predictions):
    fruit_type = "Pomme" if prediction == 1 else "Orange"
    print(f"Fruit {fruit} est probablement une {fruit_type}.")
