from torch.utils.tensorboard import SummaryWriter
import os
from socket import *
import subprocess


def pingit(host, port):  # defining function for later use

    s = socket(AF_INET, SOCK_STREAM)  # Creates socket

    try:
        s.connect((host, port))  # tries to connect to the host
    except ConnectionRefusedError:  # if failed to connect
        s.close()  # closes socket, so it can be re-used
        return False

    while True:  # If connected to host
        s.close()  # closes socket just in case
        return True


class MySummaryWriter(SummaryWriter):
    def __init__(self, numb_batches: int, base_logdir: str = os.path.join("..", "logs"),
                 experiment_name: str = "FaceRecogniction", run_name: str = "experiment1", epoch=0,
                 batch_size=8):
        self.epoch = epoch
        self.numb_batches = numb_batches
        self.base_logdir = base_logdir
        self.logpath = os.path.join(base_logdir, experiment_name, run_name)
        self.batch_size = batch_size
        self.start_tensorboard()
        self.writer = SummaryWriter(self.logpath)

    def start_tensorboard(self, host="localhost", port=6006):
        """
        If host and port cannot be pinged, this method starts tensorboard for the duration of the execution of this python programm
        :param host:
        :param port:
        """
        if pingit(host, port):
            print("tensorboard is up: http://" + host + ":" + str(port))
        else:
            print("starting tensorboard...")
            subprocess.Popen("tensorboard --logdir " + self.base_logdir)

            print("tensorboard is temporary up: http://" + host + ":" + str(
                port) + "\nWill be closed as soon as the python code exits. To run tensorboard independently execute '" + "tensorboard --logdir " + self.base_logdir + "'")

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
