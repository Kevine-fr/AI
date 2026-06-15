import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import time

initial_time = time.time()

inputs = np.array([[0 , 0 ] , [4 , 2] , [10 , 301] , [51 , 1.5] , [-4 , 2] , [0 , 2] , [2 , 4] , [2004 , -46.78] , [62 , 212034] , [1.5 , 51] , [704 , 0.45] , [4 , -203] , [-100 , 100] , [5 , -4.25] , [10 , 20] , [40400 , 20200]])
outputs = np.array([[0] , [6] , [311] , [52.5] , [-2] , [2] , [6] , [1957.22] , [212096] , [52.5] , [704.45] , [-199] , [0] , [0.75] , [30] , [60600]])

# Initialisation du model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_shape = [2] , units = 16),
])

# Ajout de données de validation (artificielles)
val_inputs = np.array([[6, 7], [7, 8], [8, 9]], dtype=float)
val_outputs = np.array([[13], [15], [17]], dtype=float)

# Compilation et optimization du model

model.compile(optimizer = 'adam' , loss = 'mean_squared_error')

# Apprentissage du model

model.fit(inputs , outputs , epochs = 50000 , verbose = 0)

print(f"\nL'apprentissage a duré {(time.time() - initial_time):.2f}s\n")

# Prediction du model

while True:

    x = float(input("Entrée la 1e valeur : "))
    y = float(input("Entrée la 2e valeur : "))

    prediction = model.predict(np.array([[x,y]]))

    print(f"\n{x} + {y} = {prediction[0][0]:.2f}")


    

