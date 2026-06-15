import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore

# Données d'entraînement
inputs = np.array([
    [0, 0], [4, 2], [10, 301], [51, 1.5], [-4, 2], [0, 2],
    [2, 4], [2004, -46.78], [62, 212034], [1.5, 51], [704, 0.45],
    [4, -203], [-100, 100], [5, -4.25], [10, 20], [40400, 20200]
])
outputs = np.array([
    [0], [6], [311], [52.5], [-2], [2], [6], [1957.22], [212096],
    [52.5], [704.45], [-199], [0], [0.75], [30], [60600]
])

# Normalisation des données z = w1x1 + w2x2 + ... + wnxn + b
# b = ?
# w1 et w2 = ?

scaler_inputs = MinMaxScaler()
scaler_outputs = MinMaxScaler()

normalized_inputs = scaler_inputs.fit_transform(inputs)
normalized_outputs = scaler_outputs.fit_transform(outputs)

# Initialisation du modèle  
model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_shape=[2] , units=16),
])

# Compilation et optimisation 
model.compile(optimizer='adam', loss='mean_squared_error')

print("Entraînement du modèle...")

# Entraînement du modèle
model.fit(normalized_inputs, normalized_outputs, epochs=10000, verbose=0)

print("Entraînement terminé !\n")

# Utilisation du modèle
print("Testez le modèle avec vos propres nombres.")

while True:
    try:
        # Lecture des valeurs d'entrée
        x = float(input("Entrez la première valeur : "))
        y = float(input("Entrez la deuxième valeur : "))

        # Normaliser les nouvelles données
        normalized_input = scaler_inputs.transform([[x, y]])

        # Prédiction
        normalized_prediction = model.predict(normalized_input, verbose=0)
        prediction = scaler_outputs.inverse_transform(normalized_prediction)

        print(f"\nRésultat prédit : {x} + {y} = {prediction[0][0]:.2f}\n") 
        
    except ValueError:
        print("Entrée invalide. Veuillez entrer des nombres valides.")
    except KeyboardInterrupt:
        print("\nFin du programme.")
        break
