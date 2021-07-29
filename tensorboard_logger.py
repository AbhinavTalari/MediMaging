import os
import shutil
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogs:
    def __init__(self, args):
        folder = f"{args.saved_models}/TensorBoard_logs"
        # if os.path.exists(folder):
        #    shutil.rmtree(folder)
        self.writer = SummaryWriter(folder, flush_secs=10)

    def plot_current_losses(self, losses=None, step=1, individual=True):
        if individual:
            for key, value in losses.items():
                self.writer.add_scalar(key, value, step)
        else:
            self.writer.add_scalars("Training Losses", losses, step)
