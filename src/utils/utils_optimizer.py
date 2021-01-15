# General Packages
import enum
import typing

# Torch Packages
from torch import optim


# ENUMS as an auxiliary support to avoid spelling issues
class CustomOptimizer(enum.Enum):
    SGD = ['sgd', 'SGD']
    ADAM = ['adam', 'Adam', 'ADAM']
    Adagrad = ['adagrad', 'Adagrad', 'ADAGRAD']
    RMSprop = ['rmsprop', 'RMSprop', 'RMSPROP']


def get_default_optimizer(model_params):
    """
    Creates an Adam optimizer object with predefined default settings.

    :param model_params: Model parameter.
    :return: Returns default optimizer - Adam -
    object with default parameter (learning_rate=1e-3, weight_decay=5e-5).
    """
    return optim.Adam(model_params, lr=1e-3, weight_decay=5e-2)


def get_optimizer(optimizer: typing.Union[str, 'CustomOptimizer'], optimizer_params,
                  optimizer_args):
    """
    Creates a custom optimizer with custom optimizer arguments.

    :param optimizer: Name of the optimizer, e.g. 'Adam', 'SGD', ...
    :param optimizer_params: Model (Network) parameters.
    :param optimizer_args: Custom parameters like learning_rate, weight_decay, etc.
    :return: Selected optimizer with custom parameters.
    """
    if (optimizer == CustomOptimizer.SGD) or (optimizer in CustomOptimizer.SGD.value):
        return optim.SGD(params=optimizer_params,
                         lr=optimizer_args['learning_rate'],
                         momentum=optimizer_args['momentum'],
                         weight_decay=optimizer_args['weight_decay'],
                         )
    elif (optimizer == CustomOptimizer.Adagrad) or (optimizer in CustomOptimizer.Adagrad.value):
        return optim.Adagrad(params=optimizer_params,
                             lr=optimizer_args['learning_rate'],
                             weight_decay=optimizer_args['weight_decay'],
                             )
    elif (optimizer == CustomOptimizer.ADAM) or (optimizer in CustomOptimizer.ADAM.value):
        return optim.Adam(params=optimizer_params,
                          lr=optimizer_args['learning_rate'],
                          betas=optimizer_args['betas'],
                          weight_decay=optimizer_args['weight_decay']
                          )
    elif (optimizer == CustomOptimizer.RMSprop) or (optimizer in CustomOptimizer.RMSprop.value):
        return optim.RMSprop(params=optimizer_params,
                             lr=optimizer_args['learning_rate'],
                             momentum=optimizer_args['momentum'],
                             alpha=optimizer_args['alpha'],
                             weight_decay=optimizer_args['weight_decay'],
                             )