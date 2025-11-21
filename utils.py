import tempfile
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.utils import load_img, img_to_array
import cv2
from PIL import Image

# model = tf.keras.models.load_model('modelv1.keras')   # premier modele entraine
model = tf.keras.models.load_model('modelv2.keras')     # modèle entraine avec un tri plus sélectif des donnees

# définir les constantes
IMG_W, IMG_H = 224, 224

def process_video(video_file):
    # Sauvegarder la video téléversee temporairement pour pouvoir la lire avec OpenCV
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(video_file.read())
    temp_file.close()

    # Lire la video en utilisant OpenCV
    cap = cv2.VideoCapture(temp_file.name)

    # liste pour stocker les différents frames du video
    video_array = []
    # traitement de la video frame par frame
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # convertir le frame en image PIL
            pil_image = Image.fromarray(frame)

            # réduire la resolution de l'image à celle attendue par le modèle
            resized_image = pil_image.resize((224, 224))

            # convertir l'image PIL en un numpy array
            resized_array = np.array(resized_image)

            # Verification du nouveau shape
            print("New shape:", resized_array.shape)

            # redimensionner et normaliser chaque frame
            resized_array = process_image(resized_array)

            # ajouter le frame sur la liste
            video_array.append(resized_array)

    cap.release()

    return video_array

def process_fichier(fichier):
    print('fichier: ', fichier)

    # To read image file buffer as a PIL Image:
    img = load_img(fichier, target_size=(IMG_W, IMG_H))

    # To convert PIL Image to numpy array:
    img_array = img_to_array(img)

    img_array = process_image(img_array)

    return img_array

def process_image(img_array):

    # Verifier le type de img_array
    print('type(img_array)', type(img_array))

    img_array = img_array / 255.
    print('shape before: ', img_array.shape)
    img_array = img_array.reshape(1, IMG_W, IMG_H, 3)  # .reshape(1, IMG_W, IMG_H, 3)
    print('shape after: ', img_array.shape)

    return img_array

def predict(img_array):
    # Prediction avec le modèle charge
    pred = model.predict(img_array)

    print('pred = ', pred)

    return pred

def afficher_prediction(prediction):
    # if y_pred[0][0] >=0.5 :
    # prediction = round(prediction[0][0] * 100)
    # if prediction >= 60:  # >=0.6 pour prendre en compte une marge d'erreur sur les donnees
    if prediction >= 0.6:  # >=0.6 pour prendre en compte une marge d'erreur sur les donnees
        # name_pred = "pleine"
        st.error(f"Le modèle a prédit que la poubelle est pleine.")

    else:
        # name_pred = "pas encore pleine"
        st.success(f"Le modèle a prédit que la poubelle n'est pas encore pleine.")
