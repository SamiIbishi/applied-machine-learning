# Torch Packages
from torch.nn import CrossEntropyLoss
from torch import optim


# Template to modify
class CNNModelTrainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, device='cpu',
                 optimizer=None, optimizer_args={},
                 loss_func=CrossEntropyLoss()):

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.device = device
        self.optimizer_args = optimizer_args
        self.optimizer = optimizer
        self.loss_func = loss_func

        self.epoch = 0

        # if the optimizer is not initialzed
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(**optimizer_args)

        if self.device == 'cuda':
            self.model.cuda()

    def train_epoch(self):
        pass

    def evaluate(self):
        pass

    def train(self, n_epochs):
        for e in n_epochs:
            self.train_epoch()
            self.evaluate()

    def inference(self, loader=None):

        # if loader is none, use self.test_loader
        # if self.test_loader is null, break

        pass
