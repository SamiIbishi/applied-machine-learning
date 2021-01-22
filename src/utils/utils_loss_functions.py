# General Packages
import enum
import typing

# Torch Packages
from torch.nn import TripletMarginLoss


# ENUMS as an auxiliary support to avoid spelling issues
class CustomLossFunctions(enum.Enum):
    TripletMarginLoss = ['triplet', 'TRIPLET', 'TripletLoss', 'TripletMarginLoss']


def get_default_loss_function():
    """
    Creates a loss function object with predefined default settings.

    :return: Returns loss_func - TripletMarginLoss -
    object with default parameters (margin=10.0, p=2, reduction='sum').
    """
    return TripletMarginLoss(margin=20.0, p=2, reduction='sum')


def get_loss_function(loss_func: typing.Union[str, 'CustomLossFunctions'], **loss_func_args):
    """
    Create a custom loss function with custom arguments.

    :param loss_func: Name of the loss function, e.g. 'TripletMarginLoss', ...
    :param loss_func_args: Custom parameters like margin, p (norm degree), reduction, etc.
    :return: Selected loss function with custom parameters.
    """
    if (loss_func == CustomLossFunctions.TripletMarginLoss) \
            or (loss_func in CustomLossFunctions.TripletMarginLoss.value):
        return TripletMarginLoss(margin=loss_func_args['margin'],
                                 p=loss_func_args['norm_degree'],
                                 reduction=loss_func_args['margin']
                                 )
