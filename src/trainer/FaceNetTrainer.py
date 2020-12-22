# Torch Packages
import torch
import torch.nn.functional as f

# General Packages
import time
import json
import typing

# Utilities
from src.utils.mytensorboard import MySummaryWriter
from src.utils.utils_optimizer import CustomOptimizer, get_optimizer, get_default_optimizer
from src.utils.utils_loss_functions import CustomLossFunctions, get_loss_function, get_default_loss_function
#import src.utils.utils_images as img_util


# Template to modify
class SiameseNetworkTrainer:
    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
            test_loader = None,
            tensorboard_writer: MySummaryWriter = None,
            optimizer: typing.Any = None,
            optimizer_args: typing.Optional[typing.Dict] = None,
            loss_func: typing.Any = None,
            loss_func_args: typing.Optional[typing.Dict] = None,
            device: str = 'cpu',
    ):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Model
        self.model = model

        # Computation device [CPU / GPU]
        if device == 'cuda' and torch.cuda.is_available():
            self.device = device
            self.model.cuda()

        # Hyperparameter - Optimizer
        if isinstance(optimizer, str) or isinstance(optimizer, CustomOptimizer):
            if optimizer_args is None:
                raise ValueError(f'Arguments dictionary for custom optimizer is missing.')
            self.optimizer = get_optimizer(optimizer, self.model.parameters(), **optimizer_args)
        elif optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = get_default_optimizer(self.model.parameters())     # default optimizer

        # Hyperparameter - Loss Function
        self.lambda1 = 0.5
        self.lambda2 = 0.01
        if isinstance(loss_func, str) or isinstance(loss_func, CustomLossFunctions):
            if loss_func_args is None:
                raise ValueError(f'Arguments dictionary for custom loss function is missing.')
            self.loss_func = get_loss_function(loss_func, **loss_func_args)
        elif loss_func:
            self.loss_func = loss_func
        else:
            self.loss_func = get_default_loss_function()   # default loss function

        # write to tensorboard
        if tensorboard_writer:
            self.tensorboard_writer = tensorboard_writer
            self.tensorboard_writer.increment_epoch()

    def train_epoch(self, epoch) -> None:
        """
        Training function for an epoch. Including loss and accuracy calculation.

        :param epoch: Current epoch.
        :return: None
        """
        print(f"####### EPOCH {epoch} Start #################################################################")

        start_time = time.time()
        log_frequency = 5

        running_loss = 0
        for batch_idx, (images, _) in enumerate(self.train_loader):
            # Get input from data loader
            anchor, positive, negative = images

            # Push tensors to GPU if available
            if torch.cuda.is_available():
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

            # Clear gradients before calculating loss
            # especially before loss.backward() is called, otherwise gradients will be accumulated
            self.optimizer.zero_grad()

            # Extract image embedding via model output
            anchor_output, positive_output, negative_output = self.model.forward(anchor, positive, negative)

            # Calculate loss
            # # LossFunc
            triplet_loss = self.loss_func(anchor_output, positive_output, negative_output)
            # # (L1 + L2) Regularization loss
            all_image_embedding_params = torch.cat([x.view(-1) for x in self.model.image_embedding.parameters()])
            l1_regularization = self.lambda1 * torch.norm(all_image_embedding_params, 1)
            l2_regularization = self.lambda2 * torch.norm(all_image_embedding_params, 2)
            # # Sum all losses
            loss = triplet_loss + l1_regularization + l2_regularization
            loss.backward()
            running_loss += loss.item()

            # Optimize model parameter
            self.optimizer.step()

            # Logging and tensorboard
            if batch_idx % log_frequency == log_frequency-1:
                print(f"[{epoch}/{self.epochs}][{batch_idx}/{len(self.train_loader)}] => running loss: {running_loss}")
                if self.tensorboard_writer:
                    self.tensorboard_writer.log_training_loss(running_loss/log_frequency, batch_idx)
                running_loss = 0

        end_time = time.time()
        print(f"####### EPOCH {epoch} DONE ####### (computation time: {end_time - start_time}) #############")

    def evaluate_epoch(self):
        """
        Evaluates the current model accuracy in the current epoch/batch.

        :return: Validation accuracy.
        """
        # switch to evaluate mode
        self.model.eval()

        correct_prediction = 0
        for batch_idx, (images, ids) in enumerate(self.valid_loader):
            # Get input from triplet 'images'
            anchor, positive, negative = images

            # Push tensors to GPU if available
            if torch.cuda.is_available():
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

            # Compute image embeddings
            emb_anchor, emb_positive, emb_negative = self.model.forward(anchor, positive, negative)

            # Distance between Anchor and Positive
            dist_ap = f.pairwise_distance(emb_anchor, emb_positive, p=2)

            # Distance between Anchor and Negative
            dist_an = f.pairwise_distance(emb_anchor, emb_negative, p=2)

            # Evaluation and logging
            for idx in range(len(dist_ap)):
                if self.tensorboard_writer:
                    self.tensorboard_writer.log_custom_scalar("dist_ap/eval", dist_ap[idx], batch_idx)
                    self.tensorboard_writer.log_custom_scalar("dist_an/eval", dist_an[idx], batch_idx)
                if dist_ap[idx] < dist_an[idx]:
                    correct_prediction += 1

            # if batch_idx == 0:
            #     fig = img_util.plot_classes_preds_face_recognition(images[0], ids[0], predictions)
            #     self.tensorboard_writer.add_figure("predictions vs. actuals", fig, 1)

        # Compute acc. Logging and tensorboard.
        valid_acc = (100. * correct_prediction) / len(self.valid_loader)
        print(f'Validation accuracy: {valid_acc}')
        if self.tensorboard_writer:
            self.tensorboard_writer.log_validation_accuracy(valid_acc)
        # TODO: Print some example pics to tensorboard with distances
        return valid_acc

    def train(self, epochs: int = 10) -> None:
        """
        Fit model on trainings data and evaluate on validation set.

        :param epochs: Number of trainings epochs.
        :return: None
        """
        self.epochs = epochs
        for epoch in range(1, self.epochs+1):
            self.train_epoch(epoch)
            if self.tensorboard_writer:
                self.tensorboard_writer.increment_epoch()
            self.evaluate_epoch()

    def create_anchor_embeddings(self, path_to_embedding: str = './src/saved/embeddings') -> None:
        """
        After the model is trained. A dictionary with all anchor embeddings will be created and stored.
        These embeddings will be used for evaluation and inference purposes.

        :param path_to_embedding: Path to embeddings directory.
        :return: None
        """
        anchor_embedding = dict()
        for batch_idx, (images, ids) in enumerate(self.train_loader):
            anchors, _, _ = images
            emb_anchors = self.model.forward_single(anchors)

            for idx in range(len(emb_anchors)):
                if ids[idx] in anchor_embedding.keys():
                    continue

                anchor_embedding[ids[idx]] = emb_anchors[idx]

        with open(path_to_embedding + f'/anchor_embedding_{time.time()}.json', 'w') as file:
            json.dump(anchor_embedding, file)
