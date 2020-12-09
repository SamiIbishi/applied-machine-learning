from torch.utils.data import Subset
import numpy as np


def split_train_test(dataset, val_ratio):
    """
    Args:
        dataset(Dataset): Dataset object which we want to split into train and validation subsets
        val_ratio(float): Splitting ratio of how large the validation set is
    """
    n_samples = len(dataset)

    # Randomize the indices for the subsets
    shuffled_indices = np.random.permutation(n_samples)

    # Split the indices based on the validation ratio
    validationset_inds = shuffled_indices[:int(n_samples * val_ratio)]
    trainingset_inds = shuffled_indices[int(n_samples * val_ratio):]

    # Create train and validation datasets from the indices
    train_dataset = Subset(dataset, indices=trainingset_inds)
    val_dataset = Subset(dataset, indices=validationset_inds)

    return train_dataset, val_dataset
