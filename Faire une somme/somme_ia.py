import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore
import time

# Initialisation du temps
origin_time = time.time()

# Étape 1 : Données d'entraînement
inputs = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=float)
outputs = np.array([[3], [5], [7], [9], [11]], dtype=float)

# Étape 2 : Création du modèle
start_time = time.time()
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2])
])
print(f"\nTemps de création du modèle : {(time.time() - start_time):.4f}s\n")

# Étape 3 : Compilation du modèle
start_time = time.time()
model.compile(optimizer='sgd', loss='mean_squared_error')
print(f"Temps de compilation du modèle : {(time.time() - start_time):.4f}s\n")

# Étape 4 : Entraînement du modèle
start_time = time.time()
print('Entraînement du modèle...\n')
model.fit(inputs, outputs, epochs=500, verbose=0)
# Pour epochs = 5 on a une erreur au bout de la 8e valeur entiere donnée suivant la logique croissante des données présentées en entrées & une erreur au bout de la 6e valeur de prédiction pour celle suivant la logique décroissante.


print(f"Durée de l'entraînement : {(time.time() - start_time):.4f}s\n")

# Étape 5 : Test du modèle
start_time = time.time()
print("Test du modèle...\n")

test_input = np.array([[10, 20], [7, 8], [3, 5], [6, 7]], dtype=float)
expected_outputs = np.array([[30], [15], [8], [13]], dtype=float)
predictions = model.predict(test_input)

# Calcul des erreurs
absolute_errors = np.abs(predictions - expected_outputs)
mae = np.mean(absolute_errors)
average_expected = np.mean(expected_outputs)
fiability = max(0, 100 - (mae / average_expected) * 100)

print(f"Temps de traitement pour les tests du modèle : {(time.time() - start_time):.4f}s\n")

# Affichage des résultats
for inp, pred, actual in zip(test_input, predictions, expected_outputs):
    print(f"L'IA prédit que la somme de {inp[0]} et {inp[1]} est : [{pred[0]:.2f}]")
    print(f"Résultat attendu : [{actual[0]:.2f}] \n")

print(f"\nFiabilité du modèle : {fiability:.2f}%")
print(f"Le processus d'apprentissage de l'IA a pris {(time.time() - origin_time):.2f}s !\n")

# Fonction de prédiction
def predict_sum(num1, num2):
    input_data = np.array([[num1, num2]], dtype=float)
    prediction = model.predict(input_data)
    return prediction[0][0]

# Interaction avec l'utilisateur
while True:
    try:
        number1 = float(input("Entrez le premier nombre : "))
        number2 = float(input("Entrez le deuxième nombre : "))
        predicted_sum = predict_sum(number1, number2)
        print(f"La somme prédite de {number1} et {number2} est : {predicted_sum:.2f}\n")
        
        continuer = input("Voulez-vous continuer ? (oui/non) : ").strip().lower()
        if continuer != "oui":
            print("Au revoir !")
            break
    except ValueError:
        print("Veuillez entrer des nombres valides.\n")
