import streamlit as st
from utils import process_fichier, process_video, predict, afficher_prediction

IMG_W, IMG_H = 224, 224

# Nom de l'application
st.title('Détection du niveau de remplissage des conteneurs à déchets')

with open("modelv1.keras", "rb") as fp:
    btn = st.download_button(
        label="Télécharger le modèle",
        data=fp,
        file_name="modelv1.keras",
        mime="application/octet-stream"
    )
# charger un fichier
fichier = st.file_uploader("Choisir un fichier", 
                           type=["png", "jpg", "jpeg", "mp4"],
                           help="Video ou photo",
                           label_visibility="visible",
                          )

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
