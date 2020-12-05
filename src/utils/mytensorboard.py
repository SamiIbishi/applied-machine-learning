from torch.utils.tensorboard import SummaryWriter
import os



class MySummaryWriter(SummaryWriter):
    def __init__(self, numb_batches: int, base_logdir: str = os.path.join("..", "logs"),
                 experiment_name: str = "FaceRecogniction", run_name: str = "experiment1", epoch=0,
                 batch_size=8):
        self.epoch = epoch
        self.numb_batches = numb_batches
        self.base_logdir = base_logdir
        self.logpath = os.path.join(base_logdir, experiment_name, run_name)
        self.batch_size = batch_size
        self.writer = SummaryWriter(self.logpath)


    def log_training_accuracy(self, acc, batch_index):
        index = self.epoch * self.numb_batches + batch_index
        self.writer.add_scalar('Accuracy/training', acc, index)

    def log_test_accuracy(self, acc, batch_index):
        index = self.epoch * self.numb_batches + batch_index
        self.writer.add_scalar('Accuracy/test', acc, index)

    def log_training_loss(self, loss, batch_index):
        index = self.epoch * self.numb_batches + batch_index
        self.writer.add_scalar('Loss/training', loss, index)

    def log_test_loss(self, loss, batch_index):
        index = self.epoch * self.numb_batches + batch_index
        self.writer.add_scalar('Loss/test', loss, index)

    def add_figure(self, tag, figure, batch_index, close=True, walltime=None):
        global_step = self.epoch * self.numb_batches + batch_index
        self.writer.add_figure(tag,figure,global_step,close,walltime)

    def increment_epoch(self):
        self.epoch += 1
