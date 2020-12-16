# Torch Packages
from torch.nn import TripletMarginLoss
import torch.nn.functional as F
from torch import optim
import torch
import torchvision

# General Packages
import time
import typing

from src.utils.mytensorboard import MySummaryWriter
#import src.utils.utils_images as img_util


# Template to modify
class SiameseNetworkTrainer:
    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
            test_loader,
            tensorboard_writer: MySummaryWriter,
            device: str = 'cpu',
            triplet_loss_func=TripletMarginLoss(margin=1.0, p=2),
            optimizer=None,
            optimizer_args: typing.Dict = None,
    ):

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.device = device
        self.optimizer_args = optimizer_args
        self.optimizer = optimizer
        self.triplet_loss_func = triplet_loss_func

        self.threshold = 1.0    # margin
        self.epoch = 0
        self.epochs = 10

        # write to tensorboard
        self.tensorboard_writer = tensorboard_writer
        self.tensorboard_writer.increment_epoch()

        # if the optimizer is not initialized
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(**optimizer_args)

        if self.device == 'cuda':
            self.model.cuda()

    def train_epoch(self):
        start_time = time.time()
        log_frequency = 5

        total_loss = 0
        running_loss = 0
        for batch_idx, (images, _) in enumerate(self.train_loader):
            # Get input from data loader
            anchor, positive, negative = images

            # Push tensors to GPU if available
            if torch.cuda.is_available():
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

            self.optimizer.zero_grad()

            # Extract image embedding via model output
            anchor_output, positive_output, negative_output = self.model.forward(anchor,
                                                                                 positive,
                                                                                 negative)

            # Calculate loss
            triplet_loss = self.triplet_loss_func(anchor_output, positive_output, negative_output)
            triplet_loss.backward()
            total_loss += triplet_loss.item()
            running_loss += triplet_loss.item()

            # Optimize model parameter
            self.optimizer.step()

            if batch_idx % log_frequency == log_frequency-1:
                self.tensorboard_writer.log_training_loss(running_loss/log_frequency, batch_idx)
                running_loss = 0
                print(
                    f"[{self.epoch}/{self.epochs}][{batch_idx}/{len(self.train_loader)}] => loss: {triplet_loss}")

        end_time = time.time()
        print(
            f"####### EPOCH {self.epoch + 1} DONE ####### (computation time: {end_time - start_time}) ##################")

    def evaluate(self):
        # switch to evaluate mode
        self.model.eval()

        correct_prediction = 0
        for batch_idx, (images, ids) in enumerate(self.valid_loader):
            # Get input from triplet 'images'
            anchor, positive, negative = images
            # (Nx3x512x512) => (NxEmbV) => (NxScalar)
            # Push tensors to GPU if available
            if torch.cuda.is_available():
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

            batch_size = anchor.shape[0]

            # compute image embeddings
            emb_anchor, emb_positive, emb_negative = self.model.forward(anchor, positive, negative)

            # Distance between Anchor and Positive
            dist_ap = torch.nn.functional.pairwise_distance(emb_anchor, emb_positive)

            # Distance between Anchor and Negative
            dist_an = torch.nn.functional.pairwise_distance(emb_anchor, emb_negative)

            #self.tensorboard_writer.log_custom_scalar("dist_ap/eval", dist_ap, batch_idx) TODO: is bisher noch ein Array, kein einzelner Punkt
            #self.tensorboard_writer.log_custom_scalar("dist_an/eval", dist_an, batch_idx) TODO: is bisher noch ein Array, kein einzelner Punkt
            # self.tensorboard_writer.log_custom_scalar("dist_ap/eval", dist_ap, batch_idx) TODO: is bisher noch ein Array, kein einzelner Punkt
            # self.tensorboard_writer.log_custom_scalar("dist_an/eval", dist_an, batch_idx) TODO: is bisher noch ein Array, kein einzelner Punkt

            for idx in range(len(dist_ap)):
                if (dist_ap[idx] <= self.threshold) & (dist_an[idx] >= self.threshold):
                    correct_prediction += 1

            # if batch_idx == 0:
            #     fig = img_util.plot_classes_preds_face_recognition(images[0], ids[0], predictions)
            #     self.tensorboard_writer.add_figure("predictions vs. actuals", fig, 1)

        # compute acc and log
        valid_acc = (100. * correct_prediction) / len(self.valid_loader)
        print(f'Validation accuracy: {valid_acc}')
        self.tensorboard_writer.log_validation_accuracy(valid_acc)
        # TODO: Print some example pics to tensorboard with distances
        return valid_acc

    def train(self, epochs: int = 10):
        self.epochs = epochs
        for e in range(self.epochs):
            self.epoch = e
            self.train_epoch()
            self.tensorboard_writer.increment_epoch()
            self.evaluate()

    def inference(self, anchor, positive, negative, loader=None):

        match = False

        # switch to evaluate mode
        self.model.eval()

        # Push tensors to GPU if available
        if torch.cuda.is_available():
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

        # compute image embeddings
        emb_anchor, emb_positive, emb_negative = self.model.forward(anchor, positive, negative)

        # Distance between Anchor and Positive
        dist_ap = torch.nn.functional.pairwise_distance(emb_anchor, emb_positive)

        # Distance between Anchor and Negative
        dist_an = torch.nn.functional.pairwise_distance(emb_anchor, emb_negative)

        if dist_ap <= dist_an:
            match = True

        return dist_ap, dist_an, match
