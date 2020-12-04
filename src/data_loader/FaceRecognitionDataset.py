# General Python Packages
import os
import glob
from os.path import join

# Torch Packages
from torch.utils.data import Dataset, DataLoader


class FaceRecognitionDataset(Dataset):
    """ Labeled Faces in the Wild dataset."""

    # define the constructor of this dataset object
    def __init__(self, dataset_folder):
        """
        Args:
            dataset_folder(string): Path to the main folder containing the dataset.
        """
        self.dataset_folder = dataset_folder

        # What else to do when creating the dataset in our case?
        # Possibly load the fielpaths into a list and store it
        # We can wrap that into a function

        self.read_file_paths()

        # a dict to store mapping of classes to indices since we need the classes to be numerical
        self.encode_classes()


    def read_file_paths(self):
        self.image_filenames = glob.glob(join(self.dataset_folder, "**/*.jpg"), recursive=True)

    def encode_classes(self):
        self.class_to_idx = dict()
        for filename in self.image_filenames:
            split_path = filename.split(os.sep)
            label = split_path[-2]
            self.class_to_idx[label] = self.class_to_idx.get(label, len(self.class_to_idx))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        image = Image.open(self.image_filenames[idx])

        # What about the label, can we get it from the filepath?

        # split the path into parts using the os separator,
        # take the folder name to be the class name (second last element)
        split_path = self.image_filenames[idx].split(os.sep)
        label = split_path[-2]

        # look up the dict we created in the __init__() method
        return image, self.class_to_idx[label]
