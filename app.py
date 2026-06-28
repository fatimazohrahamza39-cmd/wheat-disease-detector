import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Détecteur de rouille jaune du blé — Preuve de concept")

st.markdown("""
**Contexte du projet :** *Bacillus subtilis*, la bactérie utilisée dans notre biofertilisant 
encapsulé dans l'alginate, présente un double effet agronomique documenté dans la littérature : 
un effet promoteur de croissance (biofertilisant) et un effet de biocontrôle contre certains 
agents pathogènes fongiques.

**Rôle de cet outil :** ce modèle a été entraîné sur un jeu de données public (images de 
feuilles de blé, Kaggle), à titre de preuve de concept méthodologique. Il démontre la 
faisabilité d'un diagnostic visuel automatisé de l'état sanitaire du blé. Une évaluation 
réelle de l'effet biocontrôle de notre formulation nécessiterait des essais au champ ou en 
conditions contrôlées, avec un suivi photographique de nos propres parcelles traitées et 
témoins — donnée non disponible à ce stade du projet.
""")

st.write("Importez une photo de feuille de blé pour tester le fonctionnement du modèle (détection sain/rouille jaune).")

model = tf.keras.models.load_model("modele_wheat_yellowrust.h5")

fichier_image = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if fichier_image is not None:
    image = Image.open(fichier_image).convert("RGB")
    st.image(image, caption="Image importée", width=300)

    image_redimensionnee = image.resize((96, 96))
    tableau_image = np.array(image_redimensionnee)
    tableau_image = np.expand_dims(tableau_image, axis=0)

    prediction = model.predict(tableau_image)[0][0]

    if prediction > 0.5:
        confiance = prediction * 100
        st.error(f"Résultat : Rouille jaune détectée (confiance : {confiance:.1f}%)")
    else:
        confiance = (1 - prediction) * 100
        st.success(f"Résultat : Feuille saine (confiance : {confiance:.1f}%)")
