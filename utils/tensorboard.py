import os
from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter:
    def __init__(self, log_dir, name="train"):
        self.writer = SummaryWriter(os.path.join(log_dir,name))

    def log_step(self, name, value, step):
        self.writer.add_scalar(f"{name}/step", value, step)

    def log_epoch(self, name, value, epoch):
        self.writer.add_scalar(f"{name}/epoch", value, epoch)

    def log_image(self, name, image, step=0):
        # if image.shape[1] > image.shape[2]:
        #     image = image.permute(0, 2, 1)
        self.writer.add_image(name, image, step)

    def log_audio(self, name, audio, sample_rate, step=0):
        self.writer.add_audio(name, audio, step, sample_rate=sample_rate)

    def log_network(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)
        self.writer.close()