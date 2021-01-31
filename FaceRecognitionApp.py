# General Imports

import numpy as np
from pathlib import Path

import typing
from PIL import Image

# Project Modules
from torchvision.transforms import transforms

from src.model.FaceNet import FaceNet

# Streamlit Imports
import streamlit as st
import load_css

# Pytorch
import torch

load_css.local_css("style.css")

st.title('Face Recognition Application')

PRETRAINED_MODEL_PATH = Path("pretrained_model/model")
ANCHOR_EMBEDDING_PATH = Path("pretrained_model/anchor_embeddings")
TUTORIAL_ANCHOR = Path("demo_images/tutorial_anchor")
TUTORIAL_TEST = Path("demo_images/tutorial_test")


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Sanity Checks
    if not PRETRAINED_MODEL_PATH.is_file():
        st.error('Please check the "pretrained_model" directory. The pre-trained model is missing or are renamed. '
                 'Check "pretrained_model/README.md"!')
        return
    elif not ANCHOR_EMBEDDING_PATH.is_file():
        st.error('Please check the "pretrained_model" directory. The anchor embeddings are missing or are renamed.'
                 'Check "pretrained_model/README.md"!')
        return
    elif not TUTORIAL_ANCHOR.is_dir():
        st.error(
            'Please check the "demo_images" directory. The tutorial folder "tutorial_anchor" is missing.'
            'Check the README!')
        return
    elif not TUTORIAL_TEST.is_dir():
        st.error(
            'Please check the "demo_images" directory. The tutorial folder "tutorial_test" is missing. '
            'Check the README!')
        return

    # FaceNet Model
    OurFaceNet = load_pretrained_model()

    # Load anchor embedding
    OurFaceNet.anchor_embeddings = torch.load(ANCHOR_EMBEDDING_PATH, map_location=torch.device('cpu'))

    # Sidebar
    st.sidebar.title("Face Verification Setup")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Tutorial", "One-Shot Learning"])

    if app_mode == "About":
        get_section_about()

    elif app_mode == "Tutorial":
        get_section_tutorial(OurFaceNet)

    elif app_mode == "One-Shot Learning":
        get_section_one_shot_learning(OurFaceNet)


########################################################################################################################
# Section Getter Functions
########################################################################################################################

def get_section_about():
    # General explanation what this illustration should be used for
    '''

    ## Welcome

    This is a face verification application. It was build in the scope of an advanced practical course
    "Application Challenges for Machine Learning utilizing IBM Power Architecture" at Technical University
    Munich. Which is supported via the cooperation between TU Munich and IBM. The overall task is to build
    a robust face recognition model. Including the possibility of one shot learning. Additionally, ethical
    questions regarding face recognition should be elaborated.
    [OpenPower@TUM](https://openpower.ucc.in.tum.de/home2/education/teaching-and-practical-courses/winter-2020-2021/)

    '''

    '''

    #### Abstract

    Face recognition is an already well researched field. We use a variant of the state of the art 
    Siamese Neural Networks – the triplet network – for our approach to solve the face recognition task. To focus 
    training efforts, we use Transfer-Learning and different pretrained neural networks such as densenet161. To make
    the training harder and hence improve the accuracy of our model, we decided to implement a two-stage 
    triplet-selection for the triplet network. The resulting neural network architecture achieved a validation 
    accuracy of more than 99.5%. After having learnt to embed images, the model’s inference is used to perform 
    one-shot-learning: With one new image of a person as anchor, any future images of that person can be identified.

    '''

    '''

    #### Motivation and State of the Art

    Face recognition is a technique that has the objective to recognize and identify a person based on their facial 
    features. To accomplish this, a system takes a new image of the person to be identified and compares this visual
    information with stored images of all known people. Based on a similarity function for ‘visual overlap’ and a 
    threshold it decides if there is a match.

    Many different approaches tackle the challenges of face recognition. Classical computer vision and traditional 
    machine learning approaches use handpicked features or stochastically extracted feature but in general they can 
    not reach human level performance
    [(Turk@1991)](https://ieeexplore.ieee.org/document/139758/authors#authors),
    [(Ahonen@2004)](https://link.springer.com/chapter/10.1007/978-3-540-24670-1_36), 
    [(Chen@2013)](https://ieeexplore.ieee.org/document/6619233),
    [(Deng@2018)](https://ieeexplore.ieee.org/abstract/document/8276625).

    Deep learning techniques such as DeepFace [(Taigman@2014)](https://ieeexplore.ieee.org/document/139758/authors#authorshttps://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf)
    , which was proposed in 2014, raised the bar on accuracy. Many other approaches followed DeepFace, like FaceNet 
    [(Schroff@2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)
    , VGGFace [(Parkhi@2015)](https://ora.ox.ac.uk/objects/uuid:a5f2e93f-2768-45bb-8508-74747f85cad1/download_file?file_format=pdf&safe_filename=parkhi15.pdf&type_of_work=Conference+item)
    , VGGFace2 [(Cao@2018)](https://arxiv.org/abs/1710.08092); which pushed the performance even beyond human-level 
    performance. Thus, we decided to also implement our system in a deep learning fashion, FaceNet [(Schroff@2015)]
    (https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf).

    '''

    '''

    ## Our model

    Our deep learning approach included a triplet network. This model works with a triplet of input images: 
    An anchor, which is the reference for a person, a positive that depicts the same person, and a negative that 
    depicts a different person. This requirement results in every part of our pipeline being tailored to support 
    triplets.

    #### Input Data | Dataset

    To train our model we used a subset of the Large-scale CelebFaces Attributes (CelebA) dataset as an extensive 
    source of face images. It consists of about 200,000 annotated RGB images of celebrities that were cropped 
    around the face but features various variations in age, ethnicity, and obstructions such as sunglasses or hats. 
    For our subset, we used the first 10,000 images.

    #### Preprocessing

    As mentioned, our model needs triplets as input. We needed to create these triplets out of our dataset images. 
    To make training worthwhile and difficult we focused on two factors: First, we only considered persons with at 
    least five samples for our triplets. Second, we chose the images for the triplet to be similar, by using 
    L2-loss. With each triplet being one ‘sample’, we split the dataset on a common 90/10 ratio into 
    training/validation sets. All images are resized to 224x224 before feeding them to our model.

    '''

    st.write('')
    st.write('')
    st.image(Image.open(Path('documentation/images/AdvancedTripletSelection.png')),
             caption='Triplet with optimized selection of Anchor and Negative.', use_column_width=True)
    st.write('')
    st.write('')

    '''

    #### Architecture 

    We were inspired by the FaceNet (F. Schroff, 2015) concept and implemented core ideas of their proposal. 
    They used a series of convolutional layers and one fully connected layer to extract all relevant features. 
    Followed by a normalization into the embedding vector.

    '''

    st.write('')
    st.write('')
    st.image(Image.open(Path('documentation/images/FaceNet.png')),
             caption='Schematic visualization of the FaceNet architecture. Taken from the FaceNet paper.',
             use_column_width=True)
    st.write('')
    st.write('')

    ''' 

    To save computation effort of training, we utilized pretrained networks for the convolutional, feature 
    extracting layers. Our fully connected layers were made to fit the output of e.g. densenet161 and just 
    create the embedding of the face. We tried different combinations of pretrained models and output sizes of 
    the embedding.

    '''
    st.write('')
    st.write('')
    st.image(Image.open(Path('documentation/images/ModelComposition.png')),
             caption='Schematic visualization of our model composition.', use_column_width=False)


def get_section_tutorial(OurFaceNet):

    st.write()
    '''
    
    # Tutorial

    In this section we want to visualize how the face recognition is done. For this purpose we wil first select
    a few images of people (IDs) which will serve as the anchor images. Those images are taken from the
    'Large-scale CelebFaces Attributes' dataset [(CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
    After that, we then use a couple example images of the same people but in different settings.

    The list of variations includes the following:
    * Different backgrounds
    * Different lighting, bright, dark, color, etc.
    * Facing different angles
    * Wearing hats, accessories, etc.
    * Covering face with eyeglasses, sun-glasses, mustache, etc.
    * Different hair cuts, e.g. long hair, short hair, bangs, etc.
    * Same person in different age (young vs old)
    * Different facial expressions
    
    '''

    st.write()
    st.write()
    load_tutorial_anchor_images()
    test_images = load_tutorial_test_images()
    st.write()
    st.write()

    """

    In the inference phase, the test image is feed into the trained model. It calculates an embedding which is 
    a numerical vector that represents the person (face). This embedding should include all relevant facial features 
    in order to identify a person. The generated embedding is then used to compare it with all known people. 
    This is done pairwise (test person <-> known person) with a distance function which calculates a similarity 
    value. You are free to use any distance function that maps two vectors to one scalar value. In our case, we 
    decide to train our model with the use of the euclidean distance (L2 Norm). 

    """

    st.write()
    st.write()
    st.image(Image.open(Path('documentation/images/InferenceProcess.jpeg')),
             caption=f"Inference Process",
             use_column_width=True
             )
    st.write()
    st.write()

    '''

    '''

    if len(test_images) > 0:
        st.subheader("Inference")
        start_inference = st.button(label="Start Inference")

        if start_inference:
            col_1, col_2, col_3 = st.beta_columns(3)
            visited = [False, False]
            for img_path in test_images:
                img_tensor = transforms.ToTensor()(Image.open(img_path).resize([224, 224]))
                pred_id, msg = OurFaceNet.inference(
                    img_tensor, use_threshold=False, fuzzy_matches=False
                )

                msg = f"{msg.replace('!', ' -')} Person (ID): {pred_id}"
                temp_test_img_label = img_path.stem.split(" ")[0]

                if str(pred_id) == temp_test_img_label:
                    t = "<div><span class='highlight green'><span class='bold'>Passed</span> </span></div>"
                else:
                    t = "<div> <span class='highlight red'><span class='bold'>Failed</span></span></div>"

                if not visited[0]:
                    col_1.markdown(t, unsafe_allow_html=True)
                    col_1.image(Image.open(img_path), caption=f"{msg}", use_column_width=True)
                    visited[0] = True
                elif not visited[1]:
                    col_2.markdown(t, unsafe_allow_html=True)
                    col_2.image(Image.open(img_path), caption=f"{msg}", use_column_width=True)
                    visited[1] = True
                else:
                    col_3.markdown(t, unsafe_allow_html=True)
                    col_3.image(Image.open(img_path), caption=f"{msg}", use_column_width=True)
                    visited = [False, False]


def get_section_one_shot_learning(OurFaceNet):
    # Sidebar
    st.sidebar.header("One-Shot Learning")
    st.sidebar.write("You can add a new person (ID) to the model.\
                 Afterwards, you can test the verification with some test images.")
    st.sidebar.subheader("Add Person (ID)")
    person_name = st.sidebar.text_input("Enter name of person (ID):", max_chars=40)
    anchor_file = st.sidebar.file_uploader("Select anchor image:", type=['jpg', 'png', 'jpeg'])

    # Main page
    '''
    # One-Shot Learning
    '''

    st.subheader(f"Step 1: Add new Person!")

    '''
    Check the sidebar, in order to add a new person (ID) to 
    the database. After you entered a name and uploaded the image, it will be added and preprocessed 
    automatically into the database.     
    '''

    if person_name is not None and anchor_file is not None:
        # add anchor to known IDs
        OurFaceNet.create_anchor_embeddings(anchor_dict={person_name: anchor_file})
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
            st.write("")
            start_inference = st.button(label="Start Inference")
            if start_inference:
                col_1, col_2, col_3 = st.beta_columns(3)
                visited = [False, False]
                for img_path in test_images:
                    test_image = Image.open(img_path).resize([224, 224])
                    img_tensor = transforms.ToTensor()(test_image)

                    # Prediction
                    pred_id, msg = OurFaceNet.inference(
                        img_tensor, use_threshold=False, fuzzy_matches=False
                    )

                    temp_test_img_label = ' '.join([img_path.stem.split(" ")[0], img_path.stem.split(" ")[1]])
                    if str(pred_id) == temp_test_img_label:
                        t = "<div><span class='highlight green'><span class='bold'>Passed</span> </span></div>"
                        msg = f"{msg} Person (ID): {pred_id}"
                    else:
                        t = "<div> <span class='highlight red'><span class='bold'>Failed</span></span></div>"
                        msg = "Unknown person. No identity match found!"

                    if not visited[0]:
                        col_1.markdown(t, unsafe_allow_html=True)
                        col_1.image(test_image, caption=f"{msg}", use_column_width=True)
                        visited[0] = True
                    elif not visited[1]:
                        col_2.markdown(t, unsafe_allow_html=True)
                        col_2.image(test_image, caption=f"{msg}", use_column_width=True)
                        visited[1] = True
                    else:
                        col_3.markdown(t, unsafe_allow_html=True)
                        col_3.image(test_image, caption=f"{msg}", use_column_width=True)
                        visited = [False, False]

########################################################################################################################
# Auxiliary Functions
########################################################################################################################

def load_tutorial_anchor_images(path: typing.Union[str, Path] = TUTORIAL_ANCHOR) -> None:
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

    anchor_col_1, anchor_col_2, anchor_col_3 = st.beta_columns(3)
    visited = [False, False]
    for anchor in anchor_files:
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

    return


def load_tutorial_test_images(path: typing.Union[str, Path] = TUTORIAL_TEST):
    """

    :param path:
    :return:
    """

    test_images_path = Path(path)

    if not test_images_path.is_dir():
        st.error(f"The given path: '{path}' is not a directory. Please reenter valid path!")
        return None

    return [f for f in test_images_path.glob("./*") if f.is_file()]


def print_image(image_file):
    if image_file is not None:
        img = Image.open(image_file).resize([640, 480])
        st.image(img)


@st.cache()
def load_pretrained_model():
    model = FaceNet()
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


# Select frames based on the selection in the sidebar
@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index


########################################################################################################################
# Main Application Call
########################################################################################################################

if __name__ == '__main__':
    main()
