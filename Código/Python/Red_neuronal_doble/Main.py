import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from Preprocesado_modular import (
    cargar_csv, generar_ninguna, generar_caricia_dedo, generar_caricia_mano,
    generar_golpe_dedo, generar_golpe_mano, normalizar_ventanas
)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from Modelo import crear_modelo
from tensorflow.keras.utils import to_categorical
import os
from Entrenamiento_y_visualizacion import visualizar_secuencias, mostrar_info_dataset, preparar_datos, entrenar_modelo
import json

# Cargar y procesar datos
ruta_csv = "C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\TFG\\Código\\Python\\Detección de señal\\prueba_doble.csv"

df = cargar_csv(ruta_csv)
df_nada         = df[df['Etiqueta'] == 'ninguna']
df_caricia_dedo = df[df['Etiqueta'] == 'caricia_dedo']
df_caricia_mano = df[df['Etiqueta'] == 'caricia_mano']
df_golpe_dedo   = df[df['Etiqueta'] == 'golpe_dedo']
df_golpe_mano   = df[df['Etiqueta'] == 'golpe_mano']

X0, y0 = generar_ninguna(df_nada)
X1, y1 = generar_caricia_dedo(df_caricia_dedo)
X2, y2 = generar_caricia_mano(df_caricia_mano)
X3, y3 = generar_golpe_dedo(df_golpe_dedo)
X4, y4 = generar_golpe_mano(df_golpe_mano)

# Concatenar
X = np.concatenate([X0, X1, X2, X3, X4])
print("X shape final:", X.shape)
y = np.concatenate([y0, y1, y2, y3, y4])
le = LabelEncoder()
y_cod = le.fit_transform(y)
etiquetas = list(le.classes_)

# Mostrar información y visualizar ejemplos
mostrar_info_dataset(X, y)
X_norm = normalizar_ventanas(X)
visualizar_secuencias(X_norm, y)

# Preparar datos con codificación y separación
X_train, X_test, y_train, y_test, etiquetas = preparar_datos(X_norm, y_cod, etiquetas)

# Crear modelo
modelo = crear_modelo(input_shape=X_train.shape[1:], num_clases=y_train.shape[1])
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
historia = entrenar_modelo(modelo, X_train, y_train, X_test, y_test, path_modelo="modelos/mejor_modelo.keras")

# Guardar modelo y etiquetas
os.makedirs("modelos", exist_ok=True)
modelo.save("modelos/modelo_contacto_multicanal.keras")
print("Modelo guardado en modelos/modelo_contacto_multicanal.keras")
os.makedirs("datos", exist_ok=True)
np.save("datos/etiquetas_secuencias.npy", np.array(etiquetas))
with open("datos/label_encoder.pkl", "wb") as f:                       # Guarda como transformar el encoder en las etiquetas
    pickle.dump(le, f)
with open("modelos/historia_entrenamiento.json", "w") as f:
    json.dump(historia.history, f)
pd.DataFrame(historia.history).to_csv("modelos/historia_entrenamiento.csv", index=False)