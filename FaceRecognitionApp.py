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
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("README.md"))

    # FaceNet Model
    FaceNet = load_pretrained_model()

    # Sidebar
    st.sidebar.title("Face Verification Setup")
    st.sidebar.text(f"The pretrained model is loaded and setup from the 'pretrained_model' directory automatically. \
        You now can select the mode in which you want to use the application.")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Tutorial", "One Shot Learning"])

    if app_mode == "Tutorial":
        st.sidebar.text("Load images for tutorial...")
        anchor_files = load_tutorial_anchor_images()
        test_image_files = load_tutorial_test_images()

        st.sidebar.success("Loaded all tutorial images successfully!")
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("streamlit_app.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
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
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    model.eval()
    return model


# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def load_tutorial_anchor_images(tutorial_folder: str = "./tutorial"):
    """

    :param tutorial_folder:
    :return:
    """

    anchor_images = Path("./tutorial_images/test_images")

    anchor_files = [f.path for f in anchor_images.glob("./*") if f.is_file()]

    anchor = {}
    for index, anchor_files_path in enumerate(anchor_files):

        anchor[anchor_files_path.name] = st.file_uploader(anchor_files_path, type=['jpg', 'png', 'jpeg'])

    return anchor


def load_tutorial_test_images():
    """

    :param tutorial_folder:
    :return:
    """

    test_images = Path("./tutorial_images/test_images")

    if not test_images.is_dir():
        return None

    image_files = [f.path for f in test_images.glob("./*") if f.is_file()]

    images = []
    for index, img_path in enumerate(image_files):
        images.append(st.file_uploader(img_path, type=['jpg', 'png', 'jpeg']))

    return images


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    pass


# Select frames based on the selection in the sidebar
@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index


main()
