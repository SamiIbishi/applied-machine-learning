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

    def __init__(self, url: str, filename: str, dataset_dir: str = "../data/new_dataset/", unzip: bool = False):
        """
        Downloads a remote data set and stores it in target directory.

        :param url: The url from where to download the file
        :param filename: The name to save the downloaded file as
        :param dataset_dir: Path to an empty directory to save the dataset, default "../data/new_dataset/"
        :param unzip: If the data is zipped and needs to be unzipped, default 'False'
        """

        # Source
        self.url = url

        # Paths
        self.dataset_folder = dataset_dir
        self.filename = filename

        # Operations
        self.unzip = unzip

        # Download the file and unzip if necessary
        self._download_from_gdrive()

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
        remove(join(self.dataset_folder, self.filename))


# DatasetDownloader(dataset_dir="../data/celeba_dataset/images",
#                   url='https://drive.google.com/uc?id=1-gkTnvMb8ojsW1cFFkL4JA1CAy1xa6UH',
#                   filename="images.zip", unzip=True)
#
# DatasetDownloader(dataset_dir="../data/celeba_dataset",
#                   url='https://drive.google.com/uc?id=1Y3LkdANNDsdq_6_Vwkauz_CzUCuXrSmX',
#                   filename="labels.txt", unzip=False)