# General Packages
import json
import os
import time
import typing
import datetime

# Torch Packages
import torch
import torch.nn.functional as f

# Utilities
from src.utils.utils_tensorboard import MySummaryWriter
from src.utils.utils_optimizer import CustomOptimizer, get_optimizer, get_default_optimizer
from src.utils.utils_loss_functions import CustomLossFunctions, get_loss_function, \
    get_default_loss_function
import src.utils.utils_images as img_util


# Template to modify
class SiameseNetworkTrainer:
    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
            test_loader=None,
            epochs: int = 10,
            logs_per_epoch: int = 10,
            image_log_frequency: int = 5,
            tensorboard_writer: MySummaryWriter = None,
            optimizer: typing.Any = None,
            optimizer_args: typing.Optional[typing.Dict[str, typing.Any]]=None,
            loss_func: typing.Any = None,
            loss_func_args: typing.Optional[typing.Dict[str, typing.Any]]=None,
            device: str = 'cpu',
            anchor_dict: dict = None
    ):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.anchor_dict = anchor_dict

        # Model
        self.model = model

        # Computation device [CPU / GPU]
        if device == 'cuda' and torch.cuda.is_available():
            self.device = device
            self.model.cuda()

        # Get trainable parameters
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)

        # Hyperparameter - Epoch & log-frequency
        self.epochs = epochs
        self.log_frequency = int(len(train_loader) / logs_per_epoch)
        if self.log_frequency <= 0:
            self.log_frequency = 1
        self.image_log_frequency = image_log_frequency

        # Hyperparameter - Optimizer
        self.optimizer_args = optimizer_args
        if isinstance(optimizer, str) or isinstance(optimizer, CustomOptimizer):
            if optimizer_args is None:
                raise ValueError(f'Arguments dictionary for custom optimizer is missing.')
            self.optimizer = get_optimizer(optimizer, params_to_update, **optimizer_args)
        elif optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = get_default_optimizer(params_to_update)  # default optimizer

        # Hyperparameter - Loss Function
        self.loss_func_args = loss_func_args
        if isinstance(loss_func, str) or isinstance(loss_func, CustomLossFunctions):
            if loss_func_args is None:
                raise ValueError(f'Arguments dictionary for custom loss function is missing.')
            self.loss_func = get_loss_function(loss_func, **loss_func_args)
        elif loss_func:
            self.loss_func = loss_func
        else:
            self.loss_func = get_default_loss_function()  # default loss function

        # write to tensorboard
        if tensorboard_writer:
            self.tensorboard_writer = tensorboard_writer

    def train_epoch(self, epoch) -> None:
        """
        Training function for an epoch. Including loss and accuracy calculation.

        :param epoch: Current epoch.
        :return: None
        """
        print(5 * "#" + f" EPOCH {epoch:02d} Start - Training " + 15 * "#")

        # Set model in trainings mode
        self.model.train()

        start_time = time.time()
        running_loss = 0
        total_loss = 0

        for batch_idx, (images, _) in enumerate(self.train_loader):
            # Get input from data loader
            anchor, positive, negative = images

            # Push tensors to GPU if available
            if self.device == 'cuda':
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

            with torch.set_grad_enabled(True):
                # Clear gradients before calculating loss
                self.optimizer.zero_grad()

                # Extract image embedding via model output
                anchor_output, positive_output, negative_output = self.model.forward(anchor,
                                                                                     positive,
                                                                                     negative)

                # Calculate loss
                triplet_loss = self.loss_func(anchor_output, positive_output, negative_output)
                triplet_loss.backward()

                # Optimize model parameter
                self.optimizer.step()

            # Statistics
            running_loss += triplet_loss.item() * anchor_output.size(0)
            total_loss += triplet_loss.item() * anchor_output.size(0)

            # Logging and tensorboard
            if batch_idx % self.log_frequency == self.log_frequency - 1 or batch_idx == len(
                    self.train_loader) - 1:
                header = f"[{epoch:02d}/{self.epochs}][{batch_idx}/{len(self.train_loader)}]"
                epoch_loss = (running_loss / anchor.size(0)) / (batch_idx % self.log_frequency +1)
                print(f"{header} => running trainings loss: {epoch_loss:.2f}")
                if self.tensorboard_writer:
                    self.tensorboard_writer.log_training_loss(epoch_loss, batch_idx)
                running_loss = 0

        duration = time.time() - start_time
        minutes = round(duration // 60, 0)
        seconds = round(duration % 60, 0)
        print(5 * "#" + f" EPOCH {epoch:02d} DONE - computation time: "
                        f"{minutes}m {seconds}s " + 5 * "#")

        return total_loss

    def evaluate_epoch(self, epoch):
        """
        Evaluates the current model accuracy in the current epoch/batch.

        :return: Validation accuracy.
        """
        # switch to evaluate mode
        self.model.eval()

        correct_prediction = 0
        total_prediction = 0
        running_loss = 0
        running_dist_ap = 0
        running_dist_an = 0

        for batch_idx, (images, ids) in enumerate(self.valid_loader):
            # Get input from triplet 'images'
            anchor, positive, negative = images

            # Push tensors to GPU if available
            if self.device == 'cuda':
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

            # Compute image embeddings
            with torch.set_grad_enabled(False):
                emb_anchor, emb_positive, emb_negative = self.model.forward(anchor, positive,
                                                                            negative)

                # Calculate loss
                triplet_loss = self.loss_func(emb_anchor, emb_positive, emb_negative)

            # Statistics
            running_loss += triplet_loss.item() * emb_anchor.size(0)

            # Distance between Anchor and Positive
            dist_ap = f.pairwise_distance(emb_anchor, emb_positive, p=2)

            # Distance between Anchor and Negative
            dist_an = f.pairwise_distance(emb_anchor, emb_negative, p=2)

            # Evaluation and logging
            for idx in range(len(dist_ap)):
                total_prediction += 1
                running_dist_an += dist_an[idx]
                running_dist_ap += dist_ap[idx]
                if dist_ap[idx] < dist_an[idx]:
                    correct_prediction += 1

            # Logging and tensorboard
            if batch_idx % self.log_frequency == self.log_frequency - 1 or batch_idx == len(
                    self.valid_loader):
                header = f"[{epoch:02d}/{self.epochs}][{batch_idx}/{len(self.valid_loader)}]"

                # averaging
                epoch_loss = (running_loss / anchor.size(0)) / (batch_idx + 1)
                running_dist_an = running_dist_an / (
                            (batch_idx % self.log_frequency + 1) * anchor.size(0))
                running_dist_ap = running_dist_ap / (
                            (batch_idx % self.log_frequency + 1) * anchor.size(0))

                print(f"{header} => running validation loss: {epoch_loss:.2f}")

                if self.tensorboard_writer:
                    self.tensorboard_writer.log_custom_scalar("dist_ap/eval", running_dist_ap,
                                                              batch_idx)
                    self.tensorboard_writer.log_custom_scalar("dist_an/eval", running_dist_an,
                                                              batch_idx)

                running_dist_ap = 0
                running_dist_an = 0

            if (epoch % self.image_log_frequency == self.image_log_frequency - 1 or epoch == self.epochs)  \
                    and batch_idx == 0\
                    and self.tensorboard_writer:
                # Print the first batch of images with their distances to tensorboard

                fig = img_util.plot_images_with_distances(images=images, dist_an=dist_an,
                                                          dist_ap=dist_ap)
                self.tensorboard_writer.add_figure("eval/distances", fig, batch_idx)

        # Compute acc. Logging and tensorboard.
        valid_acc = (100. * correct_prediction) / total_prediction
        print(f'Validation accuracy: {valid_acc:.2f}%')
        if self.tensorboard_writer:
            self.tensorboard_writer.log_validation_accuracy(valid_acc)
        return valid_acc

    def train(self,
              path_to_saved: str = None,
              epochs: typing.Optional[int] = None,
              log_frequency: typing.Optional[int] = None) -> None:
        """
        Fit model on trainings data and evaluate on validation set.

        :param path_to_saved:
        :param epochs: Number of trainings epochs.
        :param log_frequency: Frequency in which information is logged.
        :return: None
        """

        self.start_time_training = time.time()

        if epochs:
            self.epochs = epochs

        if log_frequency:
            self.log_frequency = log_frequency

        for epoch in range(1, self.epochs + 1):
            epoch_loss = self.train_epoch(epoch)
            if self.tensorboard_writer:
                self.tensorboard_writer.increment_epoch()
            self.evaluate_epoch(epoch)

            if (epoch % self.image_log_frequency == self.image_log_frequency - 1 or epoch == self.epochs) \
                and self.tensorboard_writer:
                batch = iter(self.valid_loader).next()
                self.inference_to_tensorboard(batch)

            if epoch_loss < 10:
                print(
                    f"##### Interrupt training because training loss is {epoch_loss} and very good")
                break

        self.end_time_training = time.time()


        if path_to_saved:
            self.save_training(path_to_saved)

    def inference_to_tensorboard(self, batch, fuzzy_matches: bool = True):
        """
        Logs the positives of one batch to tensorboard with the prediction and the inference
        :param batch: the batch incl anchors, positives, negatives and the ids
        :param fuzzy_matches:
        :return:
        """
        (images, ids) = batch

        if self.tensorboard_writer:  # log inference on some pics
            self.model.create_anchor_embeddings(anchor_dict=self.anchor_dict)

            positives = images[1]
            predicted_ids = []
            for idx in range(len(ids)):
                true_id = ids[idx]
                image = positives[idx]
                if self.device == "cuda":
                    image = image.cuda()
                (predicted_id, comment) = self.model.inference(image, fuzzy_matches=fuzzy_matches,
                                                               use_threshold=False)
                predicted_ids.append(predicted_id)
            fig = img_util.plot_classes_preds_face_recognition(positives, ids, predicted_ids,
                                                               fuzzy_matches)
            self.tensorboard_writer.add_figure("inference", fig, 0)

    def save_training(self, path_to_saved: str = "./src/saved/trained_models/"):
        """

        :param path_to_saved:
        :return:
        """

        path = path_to_saved

        # Validate path to directory 'trained_models'
        if not os.path.exists(path):
            os.makedirs(path)

        # get date/time after model is trained
        date = datetime.datetime.now()
        trainings_dir = date.strftime('model_date_%Y_%m_%d_time_%H_%M')
        trainings_dir_path = os.path.join(path, trainings_dir)

        # Validate path to current training directory
        if not os.path.exists(trainings_dir_path):
            os.makedirs(trainings_dir_path)

        # Save model
        torch.save(self.model.state_dict(), os.path.join(trainings_dir_path, 'model'))

        duration = self.end_time_training - self.start_time_training
        minutes = round(duration // 60, 0)
        seconds = round(duration % 60, 0)

        # Save hyperparameter
        hyperparameter = {
            "date": date.strftime("%m/%d/%Y, %H:%M:%S"),
            "git_commit_id": "7ed040d",  # ToDo: manually edit,
            "optimizer": str(self.optimizer),
            "loss_func": str(self.loss_func),
            "epochs": self.epochs,
            "batches in train": len(self.train_loader),
            "batch size": len(iter(self.train_loader).next()[0][0]),
            "total_duration: ": f"{minutes} min {seconds} sec"
        }

        if self.optimizer_args:
            for opt_arg, opt_arg_value in self.optimizer_args.items():
                hyperparameter['optimizer_arg_' + opt_arg] = opt_arg_value

        if self.loss_func_args:
            for loss_func_arg, loss_func_arg_value in self.loss_func_args.items():
                hyperparameter['optimizer_arg_' + loss_func_arg] = loss_func_arg_value

      # torch.save(hyperparameter, os.path.join(trainings_dir_path, 'hyperparameter.json'))
        with open(os.path.join(trainings_dir_path, 'hyperparameter.json'), "w") as write_file:
            json.dump(hyperparameter, write_file)

        torch.save(self.model.anchor_embeddings, os.path.join(trainings_dir_path, 'anchor_embeddings'))
