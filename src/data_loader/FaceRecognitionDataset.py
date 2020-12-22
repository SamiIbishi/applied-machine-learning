# General Python Packages
import os
import glob
import typing
from os.path import join
import numpy as np
from typing import List, Any, Union

# Torch Packages
from torch.utils.data import Dataset
from torchvision import transforms

# PIL Package
from PIL import Image

# Substitute Package
from re import sub


class FaceRecognitionDataset(Dataset):
    # Dataset class for handling .jpg files in celeba dataset

    def __init__(self, dataset_dir: str, labels_path: str, image_width: int = 256,
                 image_height: int = 256):
        """
        Creates a data set object from given data set directory.

        :param dataset_dir: (Local) Path to the main folder containing the dataset
        :param labels_path: (Local) Path to the labels text file
        :param image_width:
        :param image_height:

        :return: dataset: Dataset object with resized images and their respective labels
        """

        # Paths
        self.dataset_folder = dataset_dir
        self.labels_path = labels_path

        # Image dimensions
        self.image_width = image_width
        self.image_height = image_height

        # Get all the subfolders respective to the different persons
        self.image_filepaths = [f.path for f in os.scandir(self.dataset_folder) if f.is_dir()]

        self.person_dict = self._create_person_dict()

        """DEPRECATED self.labels = self._labels_from_txt()"""
        """DEPRECATED
        # Update the image_filepaths to the pruned version
        self.pruned_filepaths = self._prune_filepaths()
        """
        # Method to tranform image to tensor
        self.to_tensor = transforms.ToTensor()

        # Get the triplets of original, similar, random
        self.triplets = self._create_triplets()

    def _create_person_dict(self) -> dict:
        """ Create a dict of person ID, anchor (first index) and positives """
        person_dict = dict()
        for index, image_subfolder in enumerate(self.image_filepaths):

            person_id = int(os.path.basename(image_subfolder).split('.')[0])

            image_list = []

            image_names = os.listdir(image_subfolder)
            image_names.sort()

            # Each image is stored as a list of ID and filepath
            for image_name in image_names:
                image_id = int(os.path.basename(image_name).split('.')[0])
                image_filepath = join(self.dataset_folder, str(person_id), image_name)
                image_list.append([person_id, image_id, image_filepath])

            # For now use the first image as anchor; later use a more complex method
            anchor = image_list[0]
            image_list.pop(0)

            person_dict[index] = [person_id, anchor, image_list]

        return person_dict

    def _create_triplets(self) -> list:
        # This is the list of all images we sample from for negatives
        all_images = []
        for index, person in self.person_dict.items():
            all_images.append(person[1])
            for positive in person[2]:
                all_images.append(positive)

        triplets = []

        for index, person in self.person_dict.items():
            for positive in person[2]:
                negative_counter = 0
                while negative_counter < 5:
                    random_index = np.random.randint(low=0, high=len(all_images))
                    # Do not use the same person as negative
                    if all_images[random_index][0] != person[0]:
                        triplets.append([person[1], positive, all_images[random_index]])
                        negative_counter += 1
                    else:
                        pass

        return triplets

    ''' DEPRECATED
    def _prune_filepaths(self) -> list:
        pruned_filepaths = []

        for filepath in self.image_filepaths:
            image_id = int(os.path.basename(filepath).split('.')[0])
            image_label = self.labels[image_id]
            # We only want to get faces that occur more than once
            if sum(value == image_label for value in self.labels.values()) > 1:
                pruned_filepaths.append(filepath)

        return pruned_filepaths
    '''

    """
    def _create_triplets(self) -> list:
        triplets: List[List[str]] = []

        for filepath in self.pruned_filepaths:
            image_id = int(os.path.basename(filepath).split('.')[0])
            image_label = self.labels[int(os.path.basename(filepath).split('.')[0])]
            # Collect all other images with the same label as the current one
            similar_images = [image for image, label in self.labels.items() if
                              label == image_label and image != image_id]
            # Create 5 random other samples as the third image of the triplet
            for similar_image in similar_images:
                similar_filepath = self._image_id_to_filepath(filepath, similar_image)
                samples = 0
                # 5 adversary examples are created for each image pair of the same person
                while samples < 5:
                    random_index = np.random.randint(low=0, high=len(self.pruned_filepaths) - 1)
                    # Get the filepath of the random image
                    random_image_filepath = self.pruned_filepaths[random_index]
                    # Get the ID of the image
                    random_image_id = int(os.path.basename(random_image_filepath).split('.')[0])
                    # We do not want to have the same person as an adversary example
                    if random_image_id not in similar_images and random_image_id != image_id:
                        # All 3 images selected so can create a new triplet
                        triplets.append([filepath, similar_filepath, random_image_filepath])
                        samples += 1

        return triplets
    """

    def _image_id_to_filepath(self, example_filepath: str, image_id: int) -> str:
        """
        Reconstructs the filepath of an image given its id.

        :param example_filepath: An example filepath of the same shape as the output filepath
        :param image_id: The id which we want to get the filepath for
        :return: str: The filepath corresponding to the image id
        """
        # Crude way to convert back to filepaths from an image ID in our setup
        example_id = os.path.basename(example_filepath).split('.')[0]
        padding_size = len(example_id)
        # Pad to have valid image_id length in filepath (leading zeros)
        image_id_string = str(image_id).zfill(padding_size)

        return example_filepath.replace(example_id, image_id_string)

    """DEPRECATED
    def _labels_from_txt(self) -> dict:
        labels = {}
        with open(self.labels_path) as labels_file:
            # Read each line as a key value pair and save it to the labels dict
            for line in labels_file:
                (image_id, target) = line.split()
                labels[int(sub(r"\D", "", image_id))] = target

        return labels
    """

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> typing.Tuple[list, str]:
        # Get the respective image triplet
        triplet = self.triplets[idx]

        triplet_images = []
        for image_information in triplet:
            filepath = image_information[2]
            # Resize all 3 images
            image = Image.open(filepath).resize([self.image_width, self.image_height])
            triplet_images.append(self.to_tensor(image))

        # Return the resized images as a list and the label of the original (first) image
        return triplet_images, triplet[0][0]


dataset = FaceRecognitionDataset(dataset_dir='../data/celeba_dataset/images',
                                 labels_path='../data/celeba_dataset/labels.txt')
images, label = dataset[100]

for image in images:
    image.show()
print(label)
print("length =",len(dataset))
