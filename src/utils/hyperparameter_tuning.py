import os
import time

import torch

from src.data_loader import DataSplitter
from src.data_loader.FaceRecognitionDataset import FaceRecognitionDataset
from src.model.FaceNet import FaceNet
from src.utils.utils_tensorboard import MySummaryWriter
from src.trainer import FaceNetTrainer

from src.utils.utils_pretrained_models import PretrainedModels
from src.utils.utils_optimizer import CustomOptimizer, get_optimizer

# Download data
'''DatasetDownloader(dataset_dir="../data/celeba_dataset/images",
                   url='https://drive.google.com/uc?id=1-gkTnvMb8ojsW1cFFkL4JA1CAy1xa6UH',
                   filename="images.zip", unzip=True)

DatasetDownloader(dataset_dir="../data/celeba_dataset",
                   url='https://drive.google.com/uc?id=1Y3LkdANNDsdq_6_Vwkauz_CzUCuXrSmX',
                   filename="labels.txt", unzip=False)'''

torch.backends.cudnn.deterministic = True

torch.manual_seed(999)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)

# Configurations
batch_size = 256  # 16
epochs = 200
dataset_path = "../../data/celeba_dataset_small/images/"
val_ratio = 0.1

learning_rates = [0.001]  # [0.001, 0.01, 0.0001]
optimizers = [(CustomOptimizer.SGD, "SGD")]  # [(CustomOptimizer.ADAM, "Adam"),
# (CustomOptimizer.RMSprop, "RMSprop"), (CustomOptimizer.SGD, "SGD"),
# (CustomOptimizer.Adagrad, "Adagrad")]
pretrained_models = [(PretrainedModels.DenseNet, "DenseNet")]
# [(PretrainedModels.ResNet, "ResNet"), (PretrainedModels.DenseNet, "DenseNet"),
# (PretrainedModels.VGG19, "VGG19")]

logs_per_epoch = 30
image_logs_frequency = 3
log_graph = False

experiment_name = "FaceNet_TripletNetwork_13"  # "FaceNet_TripletNetwork_4"

device = "cuda"
default_optimizer_params = {
    "learning_rate": 0.001,
    "momentum": 0.3,
    "alpha": 0.9,
    "betas": (0.85, 0.998),
    "weight_decay": 0.05,
}

print("Creating dataset...")
dataset = FaceRecognitionDataset(dataset_dir=dataset_path)

print("Created dataset, len:", len(dataset))
train_dataset, val_dataset = DataSplitter.split_train_test(dataset=dataset, val_ratio=val_ratio)

model = FaceNet()
print("Created model")
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=4,
                                           shuffle=True, sampler=None,
                                           collate_fn=None)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         num_workers=4,
                                         shuffle=True, sampler=None,
                                         collate_fn=None)
print("Created data loaders")

id_anchor_dict = dataset.get_personid_anchor_dict()

for pretrained_model, model_name in pretrained_models:
    for lr in learning_rates:
        default_optimizer_params["learning_rate"] = lr

        for optimizer_type, optimizer_name in optimizers:

            print(f"################# Training {model_name}, {lr} for {epochs} epochs")
            # init tensorboardwriter
            tensorboard_writer = MySummaryWriter(numb_batches=len(train_loader),
                                                 experiment_name=experiment_name,
                                                 run_name=model_name + "_" + optimizer_name +
                                                 "_" + str(lr), batch_size=batch_size)

            # init model and trainer
            model = FaceNet(pretrained_model=pretrained_model, device=device)
            optimizer = get_optimizer(optimizer=optimizer_type,
                                      optimizer_params=model.parameters(),
                                      optimizer_args=default_optimizer_params)

            if log_graph:
                # Log Model to tensorboard
                images, ids = iter(train_loader).next()
                if device == "cuda" and torch.cuda.is_available():
                    images = [images[0].cuda(), images[1].cuda(), images[2].cuda()]
                    model = model.cuda()
                tensorboard_writer.add_graph(model, images)
                time.sleep(10)
                print("Wrote model graph to tensorboard")

            trainer = FaceNetTrainer.FaceNetTrainer(
                model=model,
                train_loader=train_loader,
                valid_loader=val_loader,
                optimizer=optimizer,
                tensorboard_writer=tensorboard_writer,
                device=device,
                logs_per_epoch=logs_per_epoch,
                image_log_frequency=image_logs_frequency,
                anchor_dict=id_anchor_dict
            )

            save_path = os.path.join("..", "saved", "models")
            trainer.train(epochs=epochs, path_to_saved=save_path)
