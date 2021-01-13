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
    app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Tutorial", "One Shot Learning"])

    if app_mode == "About":
        # General explanation what this illustration should be used for
        st.header("Welcome")
        st.write("This is a face verification application. It was build ")
    elif app_mode == "Tutorial":

        st.sidebar.text("Load images for tutorial...")

        # FaceNet.create_anchor_embeddings(anchor_dict=anchor_files)
        st.write("########### TESTTESTTEST ###############")
        # anchor_selection = st.sidebar.selectbox("Select Person to verify:", list(anchor_files.keys()))
        st.sidebar.success("Loaded all tutorial images successfully!")
    elif app_mode == "One Shot Learning":
        # Sidebar
        st.sidebar.header("One-Shot Learning")
        st.sidebar.write("You can add a new person (ID) to the model.\
                     Afterwards, you can test the verification with some test images.")
        st.sidebar.subheader("Add Person (ID)")
        person_name = st.sidebar.text_input("Enter name of person (ID):", max_chars=20)
        image_files = st.sidebar.file_uploader("Select anchor image:", type=['jpg', 'png', 'jpeg'])

        # Main page
        st.write("The pretrained model is loaded and ready. ")
        st.subheader(f"Step 1: Add new Person!")
        st.write(f"""
                Check the sidebar, in order to add a new person (ID) to 
                the database. After you entered a name and uploaded the image, it will be added and preprocessed 
                automatically into the database. 
                 """)

        if person_name is not None and image_files is not None:
            st.subheader("Done. New Person Added!")
            our_image = Image.open(image_files)
            st.image(our_image, caption=f"Person (ID): {person_name}", use_column_width=True)

            st.subheader("Step 2: Select Test Images")
            st.write("Now, after getting setup you can select some random images to test out the verification system. \
                    Select a directory with all (face) images you want to verify."
                     )
            test_images_path = st.text_input("Relative Path to test directory: ")
            #####
            #
            # Get test images from directory | Print Test Image with verification
            #
            #####

            test_image_dir = Path(test_images_path)

            if test_image_dir.is_dir():
                test_image_files = [f for f in test_image_dir.glob("./*") if f.is_file()]
                for image_path in test_image_files:
                    pass
            # return self.to_tensor(Image.open(path).resize([height, width]))
            else:
                st.warning("Path to directory is missing! Please add path to test images.")

            st.subheader("Step 3: Inference")
            # And within an expander
            inference_settings_expander = st.beta_expander("Inference Settings", expanded=True)
            with inference_settings_expander:
                inference_threshold = st.slider("Inference Threshold: ", min_value=1, max_value=25, value=10)

            #####
            #
            # Use loaded test images to calculate some distance | Inference
            #
            #####
########################################################################################################################


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


# Select frames based on the selection in the sidebar
@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index


if __name__ == '__main__':
    main()
