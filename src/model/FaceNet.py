# General Packages
import typing

# Torch Packages
import torch
from torch import nn
import torch.nn.functional as f

# Utilities
from src.utils.utils_pretrained_models import PretrainedModels, get_pretrained_model

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class SiameseNetwork(nn.Module):
    def __init__(
            self,
            num_embedding_dimensions: int = 32,
            num_features: int = 1024,
            pretrained_model: typing.Union[str, 'PretrainedModels'] = PretrainedModels.DenseNet
    ):
        super(SiameseNetwork, self).__init__()

        # TODO: Check whether this custom CNN model for feature extraction will be needed in future or not!
        # Custom feature extractor, currently replaced by pretrained model.
        # Default: Input shape [NxCx512x512] => Output shape [Nx16x30x30]
        # [Input] => 4x[Conv2d => MaxPool2d => PReLU] => [Output]
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.PReLU(num_parameters=128, init=0.3),
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.PReLU(num_parameters=64, init=0.3),
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.PReLU(num_parameters=32, init=0.3),
        #     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.PReLU(num_parameters=16, init=0.3),
        #     nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.PReLU(num_parameters=8, init=0.3),
        # )

        self.num_features = num_features
        self.feature_extractor = get_pretrained_model(
            pretrained_model=pretrained_model,
            num_output_features=self.num_features
        )
        self.image_embedding = nn.Sequential(
            nn.Linear(self.num_features, 1024),
            nn.PReLU(num_parameters=1, init=0.3),
            nn.Linear(1024, num_embedding_dimensions),
        )

    def forward_single(self, x):
        """
        Propagation of one input image.
        :param x: Images tensor.
        :return: Embedding of input images.
        """
        # Image Embedding
        x = self.feature_extractor(x)
        x = x.view(-1, num_flat_features(x))
        x = self.image_embedding(x)

        return x

    def forward(self, anchor, positive, negative):
        """
        Propagation of triplets in parallel.
        :param anchor: Anchor images.
        :param positive: Images which contain the same person/face as the anchors.
        :param negative: Images which contain another person/face compared to the anchors.
        :return: Embedded images of the triplets.
        """
        anchor_output = self.forward_single(anchor)
        positive_output = self.forward_single(positive)
        negative_output = self.forward_single(negative)

        return anchor_output, positive_output, negative_output

    def inference(self, input, threshold: int = 10):
        """
        Takes the input, uses the trained model to embed it.
        :param threshold:
        :param input:
        :return:
        """

        self.threshold = threshold
        self.match = False

        # switch to evaluate mode
        self.model.eval()

        # Push tensors to GPU if available
        if torch.cuda.is_available():
            input = input.cuda()

        # compute image embeddings
        emb_input = self.model.forward_single(input)

        # Get anchor embeddings
        emb_anchors = self.load_anchor_embeddings()

        # Distance between embedded input and all embedded anchors
        dist = f.pairwise_distance(emb_anchors, emb_input)
        if min(dist) <= self.threshold:
            self.match = True

        return self.match

    def load_anchor_embeddings(self):
        pass

