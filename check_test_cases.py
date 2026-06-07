import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quitar warnings de TensorFlow

# Configuración

"""
 IMPORTANTE: Cargar el modelo que se desea probar
 - best_model.keras (versión base del modelo)
 - best_model_experimental.keras (modelo con una sexta capa densa y ajuste al learning rate) 
 - best_model_mobilenet.keras (modelo con MobileNetV2 como base)
"""
MODEL_PATH = 'best_model_mobilenet.keras'

TEST_CASES_DIR = 'test_cases'
IMAGE_SIZE = (224, 224)

metadata = joblib.load("processed_datasets/metadata.pkl")
CLASS_NAMES = metadata["class_names"]


def run_prediction():
    # Verificar que exista el modelo
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el archivo del modelo '{MODEL_PATH}'")
        return

    # Cargar el modelo
    print(f"Cargando modelo '{MODEL_PATH}'...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Verificar que exista el directorio de casos de prueba
    if not os.path.exists(TEST_CASES_DIR):
        print(f"Error: No se encontró el directorio '{TEST_CASES_DIR}'")
        return

    # Iterar sobre los archivos en el directorio de casos de prueba
    print("-" * 105)
    print(f"{'Nombre de Archivo':<35} | {'Real':<20} | {'Predicción':<20} | {'Confianza'}")
    print("-" * 105)

    filesFound = False
    correct_predictions = 0
    total_predictions = 0
    for filename in sorted(os.listdir(TEST_CASES_DIR)):
        filepath = os.path.join(TEST_CASES_DIR, filename)

        # Omitir directorios y archivos ocultos
        if os.path.isdir(filepath) or filename.startswith('.'):
            continue

        try:
            # Lógica para obtener etiqueta real del nombre del archivo
            real_label = "Desconocido"

            # Mapeo de abreviaturas usadas en test_cases a nombres de clases reales
            label_map = {
                "misc": "Miscellaneous Trash",
                "food": "Food Organics",
                "text": "Textile Trash",
                "veg": "Vegetation",
                "card": "Cardboard",
                "glas": "Glass",
                "meta": "Metal",
                "plas": "Plastic",
                "pape": "Paper",

            }

            # Primero buscar coincidencia exacta o de subcadena con nombres de clases reales
            for name in CLASS_NAMES:
                if name.lower() in filename.lower():
                    real_label = name
                    break

            # Si no se encontró, intentar usando el label map
            if real_label == "Desconocido":
                for abbr, full_name in label_map.items():
                    if abbr.lower() in filename.lower():
                        # Asegurarse de que el nombre completo existe en CLASS_NAMES
                        if full_name in CLASS_NAMES:
                            real_label = full_name
                            break

            # Preprocesamiento de la imagen
            # Se usa load_img para redimensionar al tamaño esperado por el modelo
            img = load_img(filepath, target_size=IMAGE_SIZE)
            img_array = img_to_array(img)
            # Convertir a batch (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)
            # Normalización (0-1) igual que durante el entrenamiento
            img_array = img_array / 255.0

            # Predicción
            predictions = model.predict(img_array, verbose=0)

            # Obtener el índice de la clase con mayor probabilidad
            score = predictions[0]
            class_idx = np.argmax(score)
            prediction_label = CLASS_NAMES[class_idx]
            confidence = score[class_idx] * 100

            print(
                f"{filename:<35} | {real_label:<20} | {prediction_label:<20} | {confidence:>8.2f}%")
            filesFound = True
            total_predictions += 1

            # Contar si la predicción es correcta
            if prediction_label == real_label:
                correct_predictions += 1

        except Exception as e:
            # Avisar si no se pudo procesar la imagen
            if not filename.startswith('.'):
                print(f"{filename:<35} | Error: No se pudo procesar")

    if not filesFound:
        print("No se encontraron imágenes válidas en el directorio.")
    else:
        print(f"\nPuntuación: {correct_predictions}/{total_predictions}")
    print("-" * 105)


if __name__ == "__main__":
    run_prediction()
