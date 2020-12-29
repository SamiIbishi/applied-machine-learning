# General Python Packages
import os
from os.path import join
from os.path import exists
from collections import Counter

# Gdrive Download Package
import gdown

# Zipfile Package
from zipfile import ZipFile


class DatasetDownloader:
    # Downloader Class to handle the dataset download and unzipping it

    def __init__(self, url: str, filename: str, dataset_dir: str = "./data/new_dataset/",
                 unzip: bool = False, preprocess: bool = False, number_of_positives: int = 2):
        """
        Downloads a remote data set and stores it in target directory.

        :param url: The url from where to download the file
        :param filename: The name to save the downloaded file as
        :param dataset_dir: Path to an empty directory to save the dataset, default
                "../data/new_dataset/"
        :param unzip: If the data is zipped and needs to be unzipped, default 'False'
        :param preprocess: If True rearrange the dataset to be fed into a siamese network
        :param number_of_positives: at least that many Positives per anchor required
        """

        self.number_of_positives = number_of_positives

        # Source
        self.url = url

        # Paths
        self.dataset_folder = dataset_dir
        self.filename = filename

        # Operations
        self.unzip = unzip
        self.preprocess = preprocess

        # Download the file and unzip if necessary
        self._download_from_gdrive()

        # Preprocess the dataset and adjust the directory structure
        if self.preprocess:
            self._preprocess_dataset_for_siamese()

    def _download_from_gdrive(self):
        # Download the file from google drive
        output_file = join(self.dataset_folder, self.filename)
        gdown.download(self.url, output_file, quiet=False)

        # Unzip if selected on init()
        if self.unzip:
            self._unzip()

    def _unzip(self):
        # Unzip the dataset
        with ZipFile(join(self.dataset_folder, self.filename), 'r') as zipfile:
            # Extract all the contents of the zip file in the same directory
            zipfile.extractall(path=self.dataset_folder)

        # And delete the .zip file
        os.remove(join(self.dataset_folder, self.filename))

    def _preprocess_dataset_for_siamese(self):
        """ The label file 'labels' is assumed to be in ../ relative to the given dataset path """

        # Read the label file with Person ID as key and Image IDs as a list of its values
        persons_dict = dict()
        with open(join(os.path.dirname(self.dataset_folder), "labels.txt")) as labels_file:

            for line in labels_file:
                (image_id, target) = line.split()
                if target in persons_dict:
                    # Append the new image_id to the existing array for this person
                    persons_dict[target].append(image_id)
                else:
                    # Create a new array for this person
                    persons_dict[target] = [image_id]

        # Adapt the folder structure
        for person, image_list in persons_dict.items():
            if len(image_list) > self.number_of_positives:
                # Create a new folder for this person's images
                os.makedirs(join(self.dataset_folder, person))
                # Copy each image in the list to this new directory
                for image in image_list:
                    os.replace(join(self.dataset_folder, image),
                               join(self.dataset_folder, person, image))
            # Else remove the images (too few samples)
            else:
                for image in image_list:
                    os.remove(join(self.dataset_folder, image))



def download_dataset(local: bool = True):
    """
    :param local: on a local machine only use subset of the data. on a server uses full dataset
    """

    # Create the directories if necessary
    if not os.path.exists("../data"):
        os.makedirs("../data")

    if not os.path.exists("../data/celeba_dataset"):
        os.makedirs("../data/celeba_dataset")
        os.makedirs("../data/celeba_dataset/images")

        # Subset of the dataset
        if local:
            DatasetDownloader(dataset_dir="../data/celeba_dataset",
                              url='https://drive.google.com/uc?id'
                                  '=1Y3LkdANNDsdq_6_Vwkauz_CzUCuXrSmX',
                              filename="labels.txt", unzip=False)

            DatasetDownloader(dataset_dir="../data/celeba_dataset/images",
                              url='https://drive.google.com/uc?id=1'
                                  '-gkTnvMb8ojsW1cFFkL4JA1CAy1xa6UH',
                              filename="images.zip", unzip=True, preprocess=True,
                              number_of_positives=4)

        # Full dataset
        else:
            DatasetDownloader(dataset_dir="../data/celeba_dataset",
                              url='https://drive.google.com/uc?id'
                                  '=1BEk3Lyw89zMWdCs9RT5G6bPQ5QMiEVuY',
                              filename="labels.txt", unzip=False)

            DatasetDownloader(dataset_dir="../data/celeba_dataset/images",
                              url='https://drive.google.com/uc?id'
                                  '=1Uqqt7EDq1gQp6hfOixVG8vZUtBVBMwVg',
                              filename="images.zip", unzip=True, preprocess=True,
                              number_of_positives=10)

