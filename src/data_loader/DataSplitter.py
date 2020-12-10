# Python packages
import typing
import numpy as np

# Torch Packages
from torch.utils.data import Subset


def split_train_test(dataset, val_ratio: float = 0.2) -> typing.Tuple[Subset, Subset]:
    """
    Splits the given data set and shuffles the indices in the respective subsets.

    :param dataset: Dataset object which we want to split into train and validation subsets
    :param val_ratio: Splitting ratio of how large the validation should be, default '0.2'

    :return: Tuple of shuffled data subsets - train_dataset, val_dataset
    """

    n_samples = len(dataset)

    # Randomize the indices for the subsets
    shuffled_indices = np.random.permutation(n_samples)

    # Split the indices based on the validation ratio
    training_set_indices = shuffled_indices[int(n_samples * val_ratio):]
    validation_set_indices = shuffled_indices[:int(n_samples * val_ratio)]

    # Create train and validation datasets from the indices
    train_dataset = Subset(dataset, indices=training_set_indices)
    val_dataset = Subset(dataset, indices=validation_set_indices)

    return train_dataset, val_dataset
