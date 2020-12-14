import torch
import torchvision
from utils import mytensorboard
from src.data_loader.FaceRecognitionDataset import FaceRecognitionDataset
from src.data_loader import DataSplitter
from src.model.FaceNet import SiameseNetwork
from src.trainer.FaceNetTrainer import SiameseNetworkTrainer

batch_size = 4

if __name__ == '__main__':
    to_pil_image = torchvision.transforms.ToPILImage()

    dataset = FaceRecognitionDataset(dataset_dir="../src/data/celeba_dataset/images/", labels_path="../src/data/celeba_dataset/labels.txt")
    print("Created dataset")
    train_dataset, val_dataset = DataSplitter.split_train_test(dataset=dataset, val_ratio=0.5)
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

    anchors = images[0]
    positives = images[1]
    negatives = images[2]

    # write graph of model to tensorboard
    tensorboard_writer.add_graph(model, images)


    # write sample images to tensorboard
    anchors_grid = torchvision.utils.make_grid(anchors)
    positives_grid = torchvision.utils.make_grid(positives)
    negatives_grid = torchvision.utils.make_grid(negatives)

    tensorboard_writer.add_image("anchor sample", anchors_grid)
    tensorboard_writer.add_image("negatives sample", negatives_grid)
    tensorboard_writer.add_image("positives sample", positives_grid)


    trainer.train(epochs=1)

