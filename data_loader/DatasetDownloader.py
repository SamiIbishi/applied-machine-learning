# General Python Packages
import os
from os.path import join
from os.path import exists
from os import remove

# Gdrive Download Package
import gdown

# Zipfile Package
from zipfile import ZipFile


class DatasetDownloader:
    # Downloader Class to handle the dataset download and unzipping it

    def __init__(self, dataset_folder, url, filename, zip):
        """
        Args:
            dataset_folder(string): Path to the folder to save the dataset to (must be empty)
            url(string): The url from where to download the file
            filename(string): The name to save the downloaded file as
            zip(bool): If the data is zipped and needs to be unzipped
        """
        self.dataset_folder = dataset_folder
        self.url = url
        self.filename = filename
        self.zip = zip

        # Download the file and unzip if necessary
        self.download_from_gdrive()

    def download_from_gdrive(self):
        # Download the file from google drive
        output_file = join(self.dataset_folder, self.filename)
        gdown.download(self.url, output_file, quiet=False)

        # Unzip if selected on init()
        if self.zip is True:
            self.unzip()

    def unzip(self):
        # Unzip the dataset
        with ZipFile(join(self.dataset_folder, self.filename), 'r') as zipfile:
            # Extract all the contents of the zip file in the same directory
            zipfile.extractall(path=self.dataset_folder)

        # And delete the .zip file
        remove(join(self.dataset_folder, self.filename))


if not exists("../data/celeba_dataset/images"):
    os.makedirs("../data/celeba_dataset/images")

DatasetDownloader(dataset_folder="../data/celeba_dataset/images",
                  url='https://drive.google.com/uc?id=1-gkTnvMb8ojsW1cFFkL4JA1CAy1xa6UH',
                  filename="images.zip", zip=True)
DatasetDownloader(dataset_folder="../data/celeba_dataset",
                  url='https://drive.google.com/uc?id=1Y3LkdANNDsdq_6_Vwkauz_CzUCuXrSmX',
                  filename="labels.txt", zip=False)
