# General Packages
import typing

# Torch Packages
import torch
from torch import nn
import torch.nn.functional as f
from torchvision import transforms

# PIL Package
from PIL import Image

# Utilities
from src.utils.utils_pretrained_models import PretrainedModels, get_pretrained_model


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class SiameseNetwork(nn.Module):
    def __init__(
            self,
            input_size: int = 224,
            num_embedding_dimensions: int = 4096,
            num_features: int = 4096,
            pretrained_model: typing.Union[str, 'PretrainedModels']=PretrainedModels.DenseNet,
            device: str = 'cpu'
    ):
        super(SiameseNetwork, self).__init__()

        self.input_size = input_size
        self.num_embedding_dimensions = num_embedding_dimensions
        self.num_features = num_features
        self.pretrained_model = pretrained_model
        self.anchor_embeddings = dict()
        
        self.feature_extractor = get_pretrained_model(
            pretrained_model=self.pretrained_model,
            num_output_features=self.num_features
        )
        
        self.image_embedding = nn.Sequential(
            nn.Linear(self.num_features, 4096),
            nn.BatchNorm1d(num_features=4096),
            nn.PReLU(num_parameters=1, init=0.3),
            nn.Linear(self.num_features, 4096),
            nn.BatchNorm1d(num_features=4096),
            nn.PReLU(num_parameters=1, init=0.3),
            nn.Linear(4096, num_embedding_dimensions),
            nn.Sigmoid()  # ToDo: nn.Tanh() oder nn.Softmax()
        )

        self.last_layer = "Sigmoid"

        self.device = device

    def forward_single(self, x):
        """
        Propagation of one input image.

        :param x: Images tensor.
        :return: Embedding of input images.
        """
        if self.device == "cuda" and torch.cuda.is_available():
            x = x.cuda()

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

    def inference(self, input_image, use_threshold: bool = True,
                  threshold: float = 10.0, fuzzy_matches: bool = True):
        """
        Recognize identity in input images.

        Process: Propagates the input images through the trained model to embed it. Then uses the
        emb_input to calculate the distance to all anchor embeddings. The smallest distance
        indicates the identity. Additionally a threshold can be used to make sure that unknown
        people are not wrongly identified as a known person (from the database).

        :param input_image: Image of person to be verified/recognized.
        :param use_threshold: Inference ID with dist < threshold, else smallest dist.
        :param threshold: If use_threshold=True, threshold determines all possible matches.
        :param fuzzy_matches: When fuzzy_matches=False.
        :return:
        """

        # Switch to evaluate mode
        self.eval()

        # Compute image embeddings
        input_image = input_image.unsqueeze(0)
        emb_input = self.forward_single(input_image)

        # Get match(es)
        matched_ids = None
        if use_threshold:
            if fuzzy_matches:
                matched_ids = list()
                for person_id, emb_anchor in self.anchor_embeddings.items():
                    dist = f.pairwise_distance(emb_anchor, emb_input).item()
                    if abs(dist) <= threshold:  # all ids with dists smaller than threshold

                        matched_ids.append((person_id, round(dist, 2)))
                matched_ids.sort(key=lambda x: x[1])
            else:
                for person_id, emb_anchor in self.anchor_embeddings.items():
                    dist = f.pairwise_distance(emb_anchor, emb_input).item()
                    if abs(dist) <= threshold:
                        # id with the smallest dist and smaller than threshold
                        matched_ids = person_id
        else:
            if fuzzy_matches:
                matched_ids = list()
                for person_id, emb_anchor in self.anchor_embeddings.items():
                    dist = f.pairwise_distance(emb_anchor, emb_input).item()
                    matched_ids.append(
                        (person_id, round(dist, 2)))  # all ids with dists smaller than threshold
                matched_ids.sort(key=lambda x: x[1])
            else:
                smallest_distance = float("inf")
                for person_id, emb_anchor in self.anchor_embeddings.items():
                    dist = f.pairwise_distance(emb_anchor, emb_input).item()
                    if abs(dist) < smallest_distance:
                        smallest_distance = dist
                        matched_ids = person_id  # id with with the smallest distance

        if matched_ids is None or not matched_ids:
            return "-1", "Unknown person. No identity match found!"
        elif isinstance(matched_ids, list):
            return matched_ids[:5], "Top 5 potential matches! (ordered by distance)"
        else:
            return matched_ids, "Best match!"

    def create_anchor_embeddings(self, anchor_dict,
                                 embedding_path: typing.Optional[str] = None) -> None:
        """
        Takes all anchor images and embeds them with the trained model. The created dictionary is
        add as a new member variable. Additionally, if a path is passed as an argument, the created
        dictionary will be stored as '.pt' file.

        :param anchor_dict: Dictionary with person ids as keys and the image paths as values.
        :param embedding_path: The path/filename in order to store the created anchor_embeddings.
        :return: None
        """
        for person_id, anchor_path in anchor_dict.items():
            anchor_image = Image.open(anchor_path).resize([self.input_size, self.input_size])
            anchor_image = transforms.ToTensor()(anchor_image).unsqueeze(0)
            self.anchor_embeddings[person_id] = self.forward_single(anchor_image)

        if embedding_path:
            torch.save(self.anchor_embeddings, embedding_path)

    def load_anchor_embeddings(self, embedding_path: typing.Optional[str] = None) -> None:
        """
        Loads a dictionary containing all anchor embeddings with their respective ids
        {'id': emb_tensor}.

        :param embedding_path: Path to stored '.pt' file containing reference dictionary.
        :return: None
        """

        self.anchor_embeddings = torch.load(embedding_path)
