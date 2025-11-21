from utils import *

IMG_W, IMG_H = 224, 224

model = tf.keras.models.load_model('modelv2.keras')
# model = tf.keras.models.load_model('modelv1.keras')

# Nom de l'application
st.title('Détection du niveau de remplissage des conteneurs à déchets')

# charger un fichier
fichier = st.file_uploader("Choisir un fichier", type=["png", "jpg", "jpeg", "mp4"])

# prendre une photo
enable = st.checkbox("Enable camera", key="enable")
picture = st.camera_input("Take a picture", disabled=not enable)

if picture is not None:
    fichier = picture

if fichier is not None:
    if fichier.type == "video/mp4":
        # traitement de la video
        frames_list = process_video(fichier)

        # regrouper les predictions sur une liste
        predictions = []
        for frame in frames_list:
            # print('frame type: ', type(frame))
            # print('frame shape: ', frame.shape)

            pred = predict(frame)
            predictions.append(pred)

        # afficher la moyenne des predictions de chaque frame de la video
        afficher_prediction(np.average(predictions))

        # afficher video
        st.video(fichier)

    else:
        # preparation de l'image pour la prediction
        img_array = process_fichier(fichier)

        # prediction
        prediction = predict(img_array)
        afficher_prediction(prediction)

        # affichage de l'image utilisée pour la prédiction
        st.image(fichier, use_container_width=True)

