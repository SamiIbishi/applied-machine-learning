import torch
import torchvision
from utils import mytensorboard
from src.data_loader.FaceRecognitionDataset import FaceRecognitionDataset
from src.data_loader import DataSplitter
from src.model.FaceNet import SiameseNetwork
from src.trainer.FaceNetTrainer import SiameseNetworkTrainer

batch_size = 8

if __name__ == '__main__':
    dataset = FaceRecognitionDataset(dataset_dir="../src/data/celeba_dataset/images/", labels_path="../src/data/celeba_dataset/labels.txt")
    print("Created dataset")
    train_dataset, val_dataset = DataSplitter.split_train_test(dataset=dataset)
    print("Splitted dataset")
    model = SiameseNetwork()
    print("Created model")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=True, sampler=None,
                                               collate_fn=None)
    print("Created train loader")
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=False, sampler=None,
                                               collate_fn=None)
    print("Created val_loader")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print("Created optimizer")
    tensorboard_writer = mytensorboard.MySummaryWriter(
        numb_batches=len(train_loader), batch_size=batch_size, experiment_name="SiameseNetwork")
    print("Created tensorboard_writer")
    trainer = SiameseNetworkTrainer(model=model, train_loader=train_loader, valid_loader=val_loader, test_loader=val_loader,
                                    optimizer=optimizer, tensorboard_writer=tensorboard_writer)
    print("Created trainer")

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    tensorboard_writer.add_image('_images', img_grid)
    tensorboard_writer.add_graph(net=model, images=images)
    # trainer.train(epochs=1)
