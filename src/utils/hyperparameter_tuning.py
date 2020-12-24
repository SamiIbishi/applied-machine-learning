import os

import torch

from src.data_loader import DataSplitter
from src.data_loader.FaceRecognitionDataset import FaceRecognitionDataset
from src.model.FaceNet import SiameseNetwork
from src.utils.utils_tensorboard import MySummaryWriter
from src.trainer import FaceNetTrainer

from src.utils.utils_pretrained_models import PretrainedModels

# Download data
'''DatasetDownloader(dataset_dir="../data/celeba_dataset/images",
                   url='https://drive.google.com/uc?id=1-gkTnvMb8ojsW1cFFkL4JA1CAy1xa6UH',
                   filename="images.zip", unzip=True)

DatasetDownloader(dataset_dir="../data/celeba_dataset",
                   url='https://drive.google.com/uc?id=1Y3LkdANNDsdq_6_Vwkauz_CzUCuXrSmX',
                   filename="labels.txt", unzip=False)'''


batch_size=16
learning_rates =  [0.001, 0.01, 0.0001]
pretrained_models = [(PretrainedModels.ResNet, "ResNet"), (PretrainedModels.VGG19, "VGG19"), (PretrainedModels.DenseNet, "DenseNet")]
regularizations = [] #ToDo wo können wir die einbauen?
epochs = 1
val_ratio=0.1
log_frequency = 10

dataset = FaceRecognitionDataset(dataset_dir="../../data/celeba_dataset/images/")
print("Created dataset, len:", len(dataset))
train_dataset, val_dataset = DataSplitter.split_train_test(dataset=dataset, val_ratio=val_ratio)

model = SiameseNetwork()
print("Created model")
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=4,
                                           shuffle=True, sampler=None,
                                           collate_fn=None)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         num_workers=4,
                                         shuffle=False, sampler=None,
                                         collate_fn=None)
print("Created data loaders")



for pretrained_model, name in pretrained_models:
    for lr in learning_rates:
        print(f"################# Training {name}, {lr} for {epochs} epochs")
        # init tensorboardwriter
        tensorboard_writer = MySummaryWriter(numb_batches=len(train_loader), experiment_name="FaceNet_TripletNetwork", run_name=name + str(lr), batch_size=batch_size)

        # init model and trainer
        model = SiameseNetwork(pretrained_model=pretrained_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        trainer = FaceNetTrainer.SiameseNetworkTrainer(
            model=model,
            train_loader = train_loader,
            valid_loader=val_loader,
            optimizer=optimizer,
            tensorboard_writer=tensorboard_writer,
            device="cuda",
            log_frequency=log_frequency
        )


        save_path = os.path.join("..", "saved", "models")
        trainer.train(epochs=epochs, path_to_saved=save_path)


'''
    # ToDo: 
    #Check Pipeline [DatenLaden, Training, Eval, Abspeichern(Model, Embeddings, Hyperparam-files), wieder laden, Inference,...]
    #Early stopping if loss(epoch)=0
    #Tensorboard logging
    #Clean machen
    
    Hyperparameters (Epochs, je 15)
    * BatchSize (8,16) Erstmal nur 16
    * Regularization [None, L1, L2, L1+L2] #Prio2
    * Für später: threshold
'''