# Tutorial - Pretrained Models Pytorch
# Source: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# General Packages
import enum
import typing

# Torch Packages
import torch.nn as nn
import torchvision.models as models


# ENUMS as an auxiliary support to avoid spelling issues
class PretrainedModels(enum.Enum):
    ResNet = ['resnet', 'ResNet', 'RESNET']
    VGG19 = ['vgg19', 'VGG19', 'vgg19bn', 'VGG19bn']
    DenseNet = ['densenet', 'DenseNet', 'DENSENET', 'DENSE', 'dense']


def set_parameter_requires_grad(model, feature_extracting: bool = True):
    """
    This method iterates over all parameters of the pretrained model and deactivates their
    requires_grad flag.
    Pretrained models already have trained weights. Their purpose is only to extract features.
    Therefore the gradients are not needed and need to be discarded.
    :param model: Pretrained model with pretrained parameters.
    :param feature_extracting: Purpose of the pretrained model. True if only to extract features,
    false otherwise.
    :return:
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_pretrained_model(
        pretrained_model: typing.Union[str, 'PretrainedModels'],
        num_output_features: int,
        feature_extract: bool = True,
        use_pretrained: bool = True
):
    """
    Get predefined models, pretrained or not. Handles tensor sizes of the extraction/last layers.
    :param pretrained_model: Name of the pretrained model which will be used to extract features.
    :param num_output_features: The output dimension the model needs to have in order to fit the
    rest of the model.
    :param feature_extract: True, if pretrained model only extracts features but will not be
    trained, otherwise false.
    :param use_pretrained: When model is loaded, all pretrained parameters will be loaded as
    well, when True.
    :return:
    """

    if (pretrained_model == PretrainedModels.ResNet) or \
            (pretrained_model in PretrainedModels.ResNet.value):
        """
        Model: Resnet152
        Model - Source:
        https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet152
        Paper: “Deep Residual Learning for Image Recognition”
        Paper - link: https://arxiv.org/pdf/1512.03385.pdf
        """
        feature_extractor_model = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(feature_extractor_model, feature_extract)
        num_inter_features = feature_extractor_model.fc.in_features
        feature_extractor_model.fc = nn.Linear(num_inter_features, num_output_features)

    elif (pretrained_model == PretrainedModels.VGG19) or \
            (pretrained_model in PretrainedModels.VGG19.value):
        """
        Model: VGG19bn
        Model - Source:
        https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg19_bn
        Paper:  “Very Deep Convolutional Networks For Large-Scale Image Recognition”
        Paper - Source: https://arxiv.org/pdf/1409.1556.pdf
        """
        feature_extractor_model = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(feature_extractor_model, feature_extract)
        num_inter_features = feature_extractor_model.classifier[6].in_features
        feature_extractor_model.classifier[6] = nn.Linear(num_inter_features, num_output_features)

    elif (pretrained_model == PretrainedModels.DenseNet) or \
            (pretrained_model in PretrainedModels.DenseNet.value):
        """
        DenseNet 161
        Model: DenseNet 161
        Model - Source:
        https://pytorch.org/docs/stable/_modules/torchvision/models/densenet.html#densenet161
        Paper:  "Densely Connected Convolutional Networks"
        Paper - Source: https://arxiv.org/pdf/1608.06993.pdf
        """
        feature_extractor_model = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(feature_extractor_model, feature_extract)
        num_inter_features = feature_extractor_model.classifier.in_features
        feature_extractor_model.classifier = nn.Linear(num_inter_features, num_output_features)

    else:
        raise ValueError('Wrong pretrained model / model name. '
                         'Check spelling or use predefined enum.')

    return feature_extractor_model
