# General Python Packages
import os
import glob
import typing
from os.path import join
import numpy as np
from typing import List, Any, Union
import operator
import datetime

# Torch Packages
from torch.utils.data import Dataset
from torch import Tensor
from torch import nn
from torch import unsqueeze
from torch import stack
from torchvision import transforms
import torchvision.models as models
import torch

# Matplotlib package
from matplotlib.pyplot import imshow
from matplotlib.pyplot import show

# PIL Package
from PIL import Image

# Substitute Package
from re import sub


class FaceRecognitionDataset(Dataset):
    # Dataset class for handling .jpg files in celeba dataset

    def __init__(self, dataset_dir: str, image_width: int = 224,
                 image_height: int = 224, overwrite_dicts: bool = False):
        """
        Creates a data set object from given data set directory.

        :param dataset_dir: (Local) Path to the main folder containing the dataset
        :param image_width:
        :param image_height:

        :return: dataset: Dataset object with resized images and their respective labels
        """

        # Anchor dict
        self.anchor_dict = dict()

        # Path
        self.dataset_folder = dataset_dir

        # Image dimensions
        self.image_width = image_width
        self.image_height = image_height

        # The model used for image to vec
        self.model = models.densenet161(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        # Remove last fully-connected layer
        new_classifier = nn.Sequential(*list(self.model.classifier.children())[:-2])
        self.model.classifier = new_classifier
        self.model.eval()

        # L2 loss- function (mean squared error)
        self.loss = nn.MSELoss()

        # Method to transform image to tensor
        self.to_tensor = transforms.ToTensor()

        # Get all the subfolders of the different persons
        self.image_filepaths = [f.path for f in os.scandir(self.dataset_folder) if f.is_dir()]

        self.person_dict_path = os.path.join(self.dataset_folder, ".." ,"person_dict.npy")
        self.triplets_path = os.path.join(self.dataset_folder, "..", "triplets.npy")
        self.anchor_dict_path = os.path.join(self.dataset_folder, "..", "anchor_dict.npy")

        if ((not overwrite_dicts) and (os.path.exists(self.person_dict_path)) and (os.path.exists(self.anchor_dict_path))):
            self.person_dict = np.load(self.person_dict_path, allow_pickle=True).item()
            self.anchor_dict = np.load(self.anchor_dict_path, allow_pickle=True).item()
            print("loaded persons dict from file: ", self.person_dict_path)
            print("loaded anchor dict from file: ", self.anchor_dict_path)
        else:
            self.person_dict = self._create_person_dict()
            np.save(self.person_dict_path, self.person_dict)
            np.save(self.anchor_dict_path, self.anchor_dict)
            print("saved persons dict to file: ", self.person_dict_path)

        # Get the triplets of original, similar, random
        if ((not overwrite_dicts) and (os.path.exists(self.triplets_path))):
            self.triplets = list(np.load(self.triplets_path, allow_pickle=True))
            print("loaded triplets from file: ", self.triplets_path)
        else:
            self.triplets = self._create_triplets()
            np.save(self.triplets_path, self.triplets)
            print("saved triplets dict to file: ", self.triplets_path)

    def get_anchor_dict(self):
        """ External method to retrieve the generated anchor_dict"""

        return self.anchor_dict

    def _choose_anchor_image(self, image_information_list: list) -> int:
        """
        Given a list of images with their embeddings calculate which image to choose as anchor and
        return its index
        :param image_information_list: Image list created in _create_person_dict() within one
                subdirectory
        :return: the index of the anchor image within the list
        """
        embedding_list = []
        loss_list = []
        for image_information in image_information_list:
            embedding_list.append(image_information[3])

        torch_embedding_list = stack(embedding_list)

        mean_emb = sum(torch_embedding_list) / len(torch_embedding_list)
        for embedding in torch_embedding_list:
            loss_list.append(self.loss(embedding, mean_emb))

        # Get the image with the smallest difference to the average embedding
        min_val, idx = min((min_val, idx) for (idx, min_val) in enumerate(loss_list))

        return idx

    def _calculate_embedding(self, filepath: str) -> Tensor:
        """
        Loads an image and calculates its embedding
        :param filepath: The filepath to the image to be embedded
        :return: The 4096 dimensional image embedding
        """
        # Loads image and resizes to 224x224
        image = self.load_preprocessed_image(filepath)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            image = image.cuda()

        dummy_batch = unsqueeze(image, 0)
        embedding = self.model(dummy_batch)

        return embedding

    def _create_person_dict(self) -> dict:
        """ Create a dict of person ID, anchor (first index) and positives """
        person_dict = dict()
        print(f"start anchor selection: {datetime.datetime.now()}")
        for index, image_subfolder in enumerate(self.image_filepaths):
            if index % 10 == 0:
                print(f"Anchor selection for person {index} of {len(self.image_filepaths)}")

            person_id = int(os.path.basename(image_subfolder).split('.')[0])

            image_list = []

            image_names = os.listdir(image_subfolder)
            image_names.sort()

            # Each image is stored as a list of ID, filepath and embedding
            for image_name in image_names:
                image_id = int(os.path.basename(image_name).split('.')[0])
                image_filepath = join(self.dataset_folder, str(person_id), image_name)
                embedding = self._calculate_embedding(image_filepath)
                image_list.append([person_id, image_id, image_filepath, embedding])

            # Anchor selection by minimal distance to the mean of all of a person's images
            index = self._choose_anchor_image(image_list)
            anchor = image_list[index]

            # Add the anchor to the anchor_dict (key: person ID, value:[filepath, embedding])
            self.anchor_dict[anchor[0]] = [anchor[2], anchor[3]]

            # Remove the anchor images from the list of positives
            del image_list[index]

            person_dict[person_id] = [anchor, image_list]
        print(f"finished anchor selection: {datetime.datetime.now()}")
        return person_dict

    def get_personid_anchor_dict(self):

        person_id_anchor_dict = {}

        # person dict is now of shape [anchor, [positive, positive,...]] , thus:
        for person in self.person_dict.items():
            person_id = person[0]
            anchor_path = person[1][0][2]
            person_id_anchor_dict[person_id] = anchor_path

        return person_id_anchor_dict

    def _create_triplets(self) -> list:
        # This is the list of all images we sample from for negatives
        print(f"start triplet creation: {datetime.datetime.now()}")
        triplets = []

        for index, person in self.person_dict.items():
            if index % 50 == 0:
                print(f"Triplet creation {index} of {len(self.person_dict)}")

            anchor_embedding = self.anchor_dict[index][1]
            distances = []

            # Get a dict of possible negatives for the anchor image with ascending distance
            for negative_anchor in self.anchor_dict.items():
                # Append ID and distance to the list to later select close negatives to anchor
                distances.append(
                    [negative_anchor[0],
                     self.loss(anchor_embedding, negative_anchor[1][1]).item()])
            distances.sort(key=lambda x: x[1])

            number_of_negatives = 5

            distances = distances[1:number_of_negatives + 1]

            for positive in person[1]:
                positive_embedding = positive[3]
                negative_counter = 0
                # Consider the top number_of_negatives closest anchors
                while negative_counter < number_of_negatives:
                    negative_person = self.person_dict[distances[negative_counter][0]]
                    # Choose the closest sample of this person as a negative
                    negatives_embeddings = [negative_person[0][3]]
                    loss_list = []
                    # Append all embedded samples of this negative person
                    for image_information in negative_person[1]:
                        negatives_embeddings.append(image_information[3])

                    torch_embedding_list = stack(negatives_embeddings)

                    for embedding in torch_embedding_list:
                        loss_list.append(self.loss(embedding, positive_embedding))

                    # Get the image with the smallest difference to the positive's embedding
                    min_val, idx = min((min_val, idx) for (idx, min_val) in enumerate(loss_list))
                    if idx == 0:
                        negative = negative_person[0]
                    else:
                        negative = negative_person[1][idx - 1]

                    triplets.append([person[0], positive, negative])
                    negative_counter += 1

        print(f"finished triplet creation: {datetime.datetime.now()}")
        return triplets

    def load_preprocessed_image(self, path: str, height: int = 224, width: int = 224) \
            -> Tensor:
        """
        :param path: filepath of the image we want to load
        :param height: height for resize operation
        :param width: width for resize operation
        :return: image converted to a Tensor
        """
        image = Image.open(path).resize([height, width])

        return self.to_tensor(image)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> typing.Tuple[list, str]:
        # Get the respective image triplet
        triplet = self.triplets[idx]

        triplet_images = []
        for image_information in triplet:
            filepath = image_information[2]
            # load all 3 images and preprocess them
            triplet_images.append(self.load_preprocessed_image(filepath, self.image_width,
                                                               self.image_height))

        # Return the resized images as a list and the label of the original (first) image
        return triplet_images, triplet[0][0]