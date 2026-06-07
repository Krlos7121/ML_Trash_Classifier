import tkinter as tk
import joblib
import os
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quitar warnings de TensorFlow

"""
 IMPORTANTE: Cargar el modelo que se desea probar
 - best_model.keras (versión base del modelo)
 - best_model_experimental.keras (modelo con una sexta capa densa y ajuste al learning rate) 
 - best_model_mobilenet.keras (modelo con MobileNetV2 como base)
"""
model = tf.keras.models.load_model("best_model.keras")

metadata = joblib.load("processed_datasets/metadata.pkl")
class_names = metadata["class_names"]


def preprocess_imgs(image):
    image = tf.reshape(image, [-1, 224, 224, 3])
    # Normalizar pixeles de la imagen a rango de 0 a 1
    image = tf.cast(image, tf.float32) / 255.0
    return image


def predict_image(image):
    image = preprocess_imgs(image)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return predicted_class, confidence


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Imágenes")
        self.root.geometry("450x450")

        self.label = tk.Label(root, text="Cargar una imagen para clasificar")
        self.label.pack(pady=20)

        self.button = tk.Button(
            root, text="Cargar Imagen", command=self.load_image)
        self.button.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                image = Image.open(file_path).resize((224, 224))
                image_array = np.array(image)
                predicted_class, confidence = predict_image(image_array)
                self.display_result(predicted_class, confidence)
                self.display_image(image)
            except Exception as e:
                messagebox.showerror(
                    "Error", f"No se pudo procesar la imagen: {e}")

    def display_result(self, predicted_class, confidence):
        class_names = metadata["class_names"]
        result_text = f"Predicción: {class_names[predicted_class]}\nConfianza: {confidence:.2f}"
        self.result_label.config(text=result_text)

    def display_image(self, image):
        image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
