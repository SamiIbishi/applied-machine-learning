# General Imports
import os
import typing
import urllib
import random

import numpy as np
from pathlib import Path
from PIL import Image,ImageEnhance

# Project Modules
from torchvision.transforms import transforms

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

        anchor_dict = load_tutorial_anchor_images()
        FaceNet.create_anchor_embeddings(anchor_dict=anchor_dict)
        test_images = load_tutorial_test_images()

        if len(test_images) > 0:
            st.subheader("Inference")
            start_inference = st.button(label="Start Inference")

            if start_inference:
                col_1, col_2, col_3 = st.beta_columns(3)
                visited = [False, False]
                for img_path in test_images:
                    img_tensor = transforms.ToTensor()(Image.open(img_path).resize([224, 224]))
                    pred_id, msg = FaceNet.inference(
                        img_tensor, use_threshold=True, threshold=0.2, fuzzy_matches=False
                    )

                    msg = f"{msg} Person (ID): {pred_id}" if pred_id != "-1" else msg
                    if not visited[0]:
                        col_1.image(Image.open(img_path), caption=f"{msg}", use_column_width=True)
                        visited[0] = True
                    elif not visited[1]:
                        col_2.image(Image.open(img_path), caption=f"{msg}", use_column_width=True)
                        visited[1] = True
                    else:
                        col_3.image(Image.open(img_path), caption=f"{msg}", use_column_width=True)
                        visited = [False, False]


    elif app_mode == "One Shot Learning":
        # Sidebar
        st.sidebar.header("One-Shot Learning")
        st.sidebar.write("You can add a new person (ID) to the model.\
                     Afterwards, you can test the verification with some test images.")
        st.sidebar.subheader("Add Person (ID)")
        person_name = st.sidebar.text_input("Enter name of person (ID):", max_chars=20)
        anchor_file = st.sidebar.file_uploader("Select anchor image:", type=['jpg', 'png', 'jpeg'])

        # Main page
        st.write("The pretrained model is loaded and ready. ")
        st.subheader(f"Step 1: Add new Person!")
        st.write(f"""
                Check the sidebar, in order to add a new person (ID) to 
                the database. After you entered a name and uploaded the image, it will be added and preprocessed 
                automatically into the database. 
                 """)

        if person_name is not None and anchor_file is not None:
            # add anchor to known IDs
            FaceNet.create_anchor_embeddings(anchor_dict={person_name: anchor_file})
            st.subheader("Done. New Person Added!")
            anchor_image = Image.open(anchor_file)
            st.image(anchor_image, caption=f"Person (ID): {person_name}", use_column_width=True)

            st.subheader("Step 2: Select Test Images")
            st.write("Now, after getting setup you can select some random images to test out the verification system. \
                    Select a directory with all (face) images you want to verify."
                     )

            test_images_path = st.text_input("Relative Path to test directory: ")
            if test_images_path == "":
                st.warning("Path to directory is missing! Please add path to test images.")
                return
            test_image_dir = Path(test_images_path)
            if test_image_dir.is_dir():
                test_images = load_tutorial_test_images(test_image_dir)
            else:
                st.warning("Path to directory is missing! Please add path to test images.")

            # Step 3: Inference
            if len(test_images) > 0:
                st.subheader("Step 3: Inference")
                st.write(f"Now we will verify the given test images w.r.t. given Person (ID): {person_name}.")
                # And within an expander
                inference_settings_expander = st.beta_expander("Inference Settings", expanded=False)
                with inference_settings_expander:
                    inference_threshold = st.slider(
                        "Inference Threshold: ", min_value=0.01, max_value=1.0, value=0.5, step=0.05)

                st.write("")
                start_inference = st.button(label="Start Inference")
                if start_inference:
                    col_1, col_2, col_3 = st.beta_columns(3)
                    visited = [False, False]
                    for img_path in test_images:
                        img_tensor = transforms.ToTensor()(Image.open(img_path).resize([224, 224]))
                        pred_id, msg = FaceNet.inference(
                            img_tensor, use_threshold=True, threshold=inference_threshold, fuzzy_matches=False
                        )
                        st.write(f"Threshold: {inference_threshold}")

                        test_image = Image.open(img_path)
                        msg = f"{msg} Person (ID): {pred_id}" if pred_id != "-1" else msg
                        # msg = f"First 5 Person (ID): {pred_id}"
                        if not visited[0]:
                            col_1.image(test_image, caption=f"{msg}", use_column_width=True)
                            visited[0] = True
                        elif not visited[1]:
                            col_2.image(test_image, caption=f"{msg}", use_column_width=True)
                            visited[1] = True
                        else:
                            col_3.image(test_image, caption=f"{msg}", use_column_width=True)
                            visited = [False, False]

########################################################################################################################


def load_tutorial_anchor_images(path: str = "./tutorial_images/anchor_images"):
    """

    :param path:
    :return:
    """

    anchor_images_path = Path(path)

    if not anchor_images_path.is_dir():
        st.error(f"The given path: '{path}' is not a directory. Please reenter valid path!")
        return None

    anchor_dict = {}
    anchor_files = [f for f in anchor_images_path.glob("./*") if f.is_file()]

    st.write("Add known Person-IDs")

    anchor_col_1, anchor_col_2, anchor_col_3 = st.beta_columns(3)
    visited = [False, False]
    for anchor in anchor_files:
        anchor_dict[anchor.stem] = anchor
        if not visited[0]:
            anchor_col_1.image(
                Image.open(anchor).resize([224, 224]), use_column_with=True, caption=f"Person (ID): {anchor.stem}")
            visited[0] = True
        elif not visited[1]:
            anchor_col_2.image(
                Image.open(anchor).resize([224, 224]), use_column_with=True, caption=f"Person (ID): {anchor.stem}")
            visited[1] = True
        else:
            anchor_col_3.image(
                Image.open(anchor).resize([224, 224]), use_column_with=True, caption=f"Person (ID): {anchor.stem}")
            visited = [False, False]
    return anchor_dict


def load_tutorial_test_images(path: str = "./tutorial_images/test_images"):
    """

    :param path:
    :return:
    """

    test_images_path = Path(path)

    if not test_images_path.is_dir():
        st.error(f"The given path: '{path}' is not a directory. Please reenter valid path!")
        return None

    image_files = [f for f in test_images_path.glob("./*") if f.is_file()]

    random_idx = random.sample(range(0, len(image_files)), 6)

    st.write("Random sampling of test images")
    col_1, col_2, col_3 = st.beta_columns(3)
    visited = [False, False]
    for idx in range(len(random_idx)):
        if not visited[0]:
            col_1.image(Image.open(image_files[idx]).resize([224, 224]), use_column_with=True)
            visited[0] = True
        elif not visited[1]:
            col_2.image(Image.open(image_files[idx]).resize([224, 224]), use_column_with=True)
            visited[1] = True
        else:
            col_3.image(Image.open(image_files[idx]).resize([224, 224]), use_column_with=True)
            visited = [False, False]
    return image_files


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
