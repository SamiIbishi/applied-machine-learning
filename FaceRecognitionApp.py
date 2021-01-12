# General Imports
import os
import urllib
import numpy as np
from pathlib import Path
from PIL import Image,ImageEnhance

# Project Modules
from src.model.FaceNet import SiameseNetwork

# Streamlit Imports
import streamlit as st

# Pytorch
import torch

st.title('Face Recognition Application')

PRETRAINED_MODEL_PATH = Path("pretrained_model/model")


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():

    # FaceNet Model
    FaceNet = load_pretrained_model()

    # Sidebar
    st.sidebar.title("Face Verification Setup")
    st.sidebar.text(f"The pretrained model is loaded and setup from the 'pretrained_model' directory automatically. \
        You now can select the mode in which you want to use the application.")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Tutorial", "One Shot Learning"])

    if app_mode == "Tutorial":
        st.sidebar.text("Load images for tutorial...")
        image_files = st.file_uploader("Select anchor image", type=['jpg', 'png', 'jpeg'])
        if image_files is not None:
            our_image = Image.open(image_files)
            st.text("Original Image")
            # st.write(type(our_image))
            st.image(our_image)

        # FaceNet.create_anchor_embeddings(anchor_dict=anchor_files)
        st.write("########### TESTTESTTEST ###############")
        # anchor_selection = st.sidebar.selectbox("Select Person to verify:", list(anchor_files.keys()))
        st.sidebar.success("Loaded all tutorial images successfully!")
    elif app_mode == "About":
        # General explanation what this illustration should be used for
        st.write("########### BLABLABLA ###############")
    elif app_mode == "Run the app":
        run_the_app()


########################################################################################################################
# return self.to_tensor(Image.open(path).resize([height, width]))


def print_image(image_file):
    if image_file is not None:
        img = Image.open(image_file).resize([640, 480])
        st.image(img)


@st.cache()
def load_pretrained_model():
    model = SiameseNetwork()
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    pass


# Select frames based on the selection in the sidebar
@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index


main()
