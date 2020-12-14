import time
import os
import subprocess
from socket import socket, AF_INET, SOCK_STREAM
import shutil

from torch.utils.tensorboard import SummaryWriter


def pingit(host: str, port: int):
    """
    Tries to set up a connection to a host and port to test if it can be pinged
    Closes the connection immediately

    :param host:
    :param port:
    :return: True, if the connection attempt was accepted and False if not.
    """

    # init socket
    s = socket(AF_INET, SOCK_STREAM)

    try:
        s.connect((host, port))  # tries to connect to the host
    except ConnectionRefusedError:  # if failed to connect
        s.close()  # closes socket, so it can be re-used
        return False

    s.close()  # closes socket just in case
    return True


class MySummaryWriter(SummaryWriter):
    """
    Personalized tensorboard writer to simplify logging to tensorboard
    """

    def __init__(self,
                 numb_batches: int,
                 base_logdir: str = os.path.join("..", "logs"),
                 experiment_name: str = "FaceRecogniction",
                 run_name: str = "run_1",
                 epoch: int = 0,
                 batch_size: int = 8,
                 overwrite_logs: bool = False
                 ):
        self.epoch = epoch
        self.numb_batches = numb_batches
        self.base_logdir = base_logdir
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.logpath = self.get_empty_logpath(overwrite_logs)
        print("logpath: ", self.logpath)
        self.batch_size = batch_size
        self.writer = SummaryWriter(self.logpath)
        self.start_tensorboard()

    def get_empty_logpath(self, overwrite_logs: bool):
        """

        :param base_logdir:
        :param experiment_name:
        :param run_name:
        :param overwrite_logs:
        :return:
        """
        logpath = os.path.join(self.base_logdir, self.experiment_name, self.run_name)
        if (not os.path.exists(logpath)) or len(os.listdir(logpath)) == 0:
            return logpath
        else:
            print("nonempty logpath")
            if (overwrite_logs):
                shutil.rmtree(logpath)
                time.sleep(5)
                return logpath
            else:
                runs = os.listdir(os.path.join(self.base_logdir, self.experiment_name))
                max = 1
                for run in runs:
                    splitted = run.split("(")
                    if len(splitted) > 1 and splitted[0] == self.run_name:
                        try:
                            val = int(splitted[-1][:-1])
                            if val >= max:
                                max = val + 1
                        except ValueError:
                            None

                logpath = logpath + "(" + str(max) + ")"
                return logpath

    def start_tensorboard(self, host="localhost", port=6006):
        """
        If host and port cannot be pinged, this method starts tensorboard for the
        duration of the execution of this python programm
        :param host:
        :param port:
        """
        if pingit(host, port):
            print("tensorboard is up: http://" + host + ":" + str(port))
        else:
            print("starting tensorboard...")
            subprocess.Popen("tensorboard --logdir " + self.base_logdir)

            print("tensorboard is temporary up: http://" + host + ":" + str(
                port) + "\nWill be closed as soon as the python code exits. To run tensorboard "
                        "independently execute '" + "tensorboard --logdir " + self.base_logdir +
                  "'")

    def log_training_accuracy(self, acc, batch_index):
        index = self.epoch * self.numb_batches + batch_index
        self.writer.add_scalar('Accuracy/training', acc, index)

    def log_test_accuracy(self, acc, batch_index):
        index = self.epoch * self.numb_batches + batch_index
        self.writer.add_scalar('Accuracy/test', acc, index)

    def log_validation_accuracy(self, acc, batch_index):
        index = self.epoch * self.numb_batches + batch_index
        self.writer.add_scalar('Accuracy/validation', acc, index)

    def log_training_loss(self, loss, batch_index):
        index = self.epoch * self.numb_batches + batch_index
        self.writer.add_scalar('Loss/training', loss, index)

    def log_test_loss(self, loss, batch_index):
        index = self.epoch * self.numb_batches + batch_index
        self.writer.add_scalar('Loss/test', loss, index)

    def log_validation_loss(self, loss, batch_index):
        index = self.epoch * self.numb_batches + batch_index
        self.writer.add_scalar('Loss/validation', loss, index)

    def add_figure(self, tag, figure, batch_index=None, close=True, walltime=None):
        if (batch_index):
            global_step = self.epoch * self.numb_batches + batch_index
        else:
            global_step = None
        self.writer.add_figure(tag, figure, global_step, close, walltime)

    def add_image(self, tag, img, batch_index=None, walltime=None, dataformats="CHW"):
        if (batch_index):
            global_step = self.epoch * self.numb_batches + batch_index
        else:
            global_step = None
        self.writer.add_image(tag, img, global_step, walltime, dataformats)

    def add_graph(self, net, images):
        self.writer.add_graph(net, images)

    def increment_epoch(self):
        self.epoch += 1
