# Torch Packages
from torch.nn import TripletMarginLoss
from torch import optim
import torch
import torchvision

# General Packages
import time
import typing


# Template to modify
class SiameseNetworkTrainer:
    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
            test_loader,
            tensorboard_writer,
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

        self.epoch = 0

        # write to tensorboard
        self.tensorboard_writer = tensorboard_writer

        # if the optimizer is not initialized
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(**optimizer_args)

        if self.device == 'cuda':
            self.model.cuda()

    def train_epoch(self, epoch, epochs):
        start_time = time.time()
        total_loss = 0
        for batch_idx, (images, ids) in enumerate(self.train_loader):
            # Get input from data loader
            anchor, positive, negative = images
            target_ids = ids

            # Push tensors to GPU if available
            if torch.cuda.is_available():
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
                target_ids = target_ids.cuda()

            # Extract image embedding via model output
            anchor_output, positive_output, negative_output = self.model.forward_triple(anchor, positive, negative)

            # Calculate loss
            triplet_loss = self.triplet_loss_func(anchor_output, positive_output, negative_output)
            triplet_loss.backward()
            total_loss += triplet_loss.item()

            # Optimize model parameter
            self.optimizer.zero_grad()
            self.optimizer.step()

            if batch_idx % 100 == 0:
                print(f"[{epoch}/{epochs}][{batch_idx*len(anchor)}/{len(self.train_loader)}] => loss: {triplet_loss}")

        end_time = time.time()
        print(f"####### EPOCH {epoch + 1} DONE ####### (computation time: {end_time - start_time}) ##################")

    def evaluate(self):
        pass

    def train(self, epochs: int = 10):
        for epoch in range(epochs):
            self.train_epoch(epoch, epochs)
            self.evaluate()

    def inference(self, loader=None):

        # if loader is none, use self.test_loader
        # if self.test_loader is null, break

        pass

