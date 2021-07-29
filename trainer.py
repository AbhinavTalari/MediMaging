from tqdm import tqdm
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import get_dataloader
from resnet import ResidualNet
from utils import load_pretrained, get_transformations
from dataloader import get_dataloader
from tensorboard_logger import TensorboardLogs
from configs import config
from sty import fg, bg, ef, rs
import numpy as np
from sklearn import metrics
from autoencoder import SegNet


from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
import warnings
import os
import logging

warnings.filterwarnings("ignore")


"""
@TODO
1) include schedulers
2) complete the train and test functions

"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer:
    """Contains methods to train and validate a model"""

    def __init__(self, args, dataloaders_dict, num_blocks):
        self.args = args
        self.train_loader = dataloaders_dict.get("train_loader")
        self.valid_loader = dataloaders_dict.get("valid_loader")
        self.test_loader = dataloaders_dict.get("test_loader")
        self.label_map = dataloaders_dict.get("label_map")
        self.freeze_prev = args.freeze_prev_layers
        self.num_blocks = num_blocks

        if not args.scratch and args.load_pretrained and num_blocks > 1:
            pretrained_model = torch.load(
                os.path.join(
                    args.saved_models,
                    "best_val_loss_{}_model.pt".format(num_blocks - 1),
                )
            )

            model = SegNet(
                num_classes=3,
                in_channels=3,
                is_unpooling=True
            )

            self.model = load_pretrained(
                model,
                pretrained_model,
                ignore_layers=["fc.weight", "fc.bias"],
                verbose=False,
                state_dict=True,
            )
            logger.info("loaded pretrained model")

        else:
            self.model = SegNet(
                num_classes=3,
                in_channels=3,
                is_unpooling=True
            )

        # if self.freeze_prev and num_blocks > 1:
        #     max_blocks = self.args.max_resnet_blocks
        #     for param in list(self.model.children())[: -2 - (max_blocks - num_blocks)]:
        #         param.requires_grad = False

        #     logger.info('Previous blocks freezed!')

        self.model.to(self.args.device)

        self.lr = args.lr
        self.optim = args.optim
        self.weight_decay = args.weight_decay

        self.optimizer = self._create_optimizer(self.lr)
        self.scheduler = self._create_scheduler()
        self.criterion = nn.MSELoss()

        self.visualizer = TensorboardLogs(args)
        self.global_train_steps = 0
        self.global_val_steps = 0
        self.global_test_steps = 0

    def _create_optimizer(self, lr):
        params = self.model.parameters()

        if self.freeze_prev and self.num_blocks > 1:
            params = list(
                eval(f"self.model.layer{self.num_blocks}.parameters()")
            ) + list(self.model.fc.parameters())

        if self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
        elif self.optim == "adadelta":
            optimizer = torch.optim.Adadelta(params, lr=lr)
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=0.9,
                dampening=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise Exception("Not a valid optimizer offered: {0}".format(self.optim))
        return optimizer

    def _create_scheduler(self):
        try:
            if self.args.scheduler == "step_lr":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=self.step_size,
                    gamma=0.1,
                    last_epoch=-1,
                )

            elif self.args.scheduler == "cyclic_lr":
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    self.optimizer, base_lr=self.lr / 10, max_lr=self.lr
                )
            else:
                raise Exception(
                    f"The scheduler, {self.scheduler} is not yet supported!"
                )

        except Exception as exc:
            logger.debug("Scheduler not set by user. Using ReduceLROnPlateau!")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=self.args.max_epochs / 10,
                min_lr=self.lr / 10,
            )
        return scheduler

    def _update_dataloaders(self, epoch_num):
        num_blocks_to_img_size_dict = config.num_blocks_to_img_size_dict
        img_size = num_blocks_to_img_size_dict.get(str(self.num_blocks))
        transforms = get_transformations(epoch_num, phase="train")

        dataloaders_dict = get_dataloader(
            self.args, img_size, train_transforms=transforms
        )

        self.train_loader = dataloaders_dict.get("train_loader")
        self.valid_loader = dataloaders_dict.get("valid_loader")
        self.test_loader = dataloaders_dict.get("test_loader")
        self.label_map = dataloaders_dict.get("label_map")

    def train_epoch(self, epoch):
        running_mse, aug_running_mse = 0, 0
        batch_idx = 0
        correct, total = 0, 0
        confusion_matrix, aug_confusion_matrix = torch.zeros(3, 3), torch.zeros(3, 3)

        all_p, all_g = [], []

        if epoch in config.steps:
            self._update_dataloaders(epoch)
            logger.info("Dataloader updated with transforms!")

        for (inputs, labels) in tqdm(self.train_loader):
            self.model.train()
            batch_idx += 1

            # fetch the data from the dataloader
            inputs = inputs.to(self.args.device)
            # augment the input samples based on epoch number
            # augmented_inputs = augment_data_by_epoch(inputs, epoch)
            labels = labels.to(self.args.device)

            out = self.model(inputs)
            loss = self.criterion(out, labels.squeeze(1).long())
            running_mse += loss.item()
            # train_loss += loss.item() * inputs.size(0)

            # backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(out.data, 1)

            all_p.append(predicted.detach().cpu().numpy())
            all_g.append(labels.squeeze(1).detach().cpu().numpy())

            total += labels.squeeze(1).size(0)
            correct += (predicted == labels.squeeze(1)).sum().item()
            for t, p in zip(labels.squeeze(1).view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            self.global_train_steps += 1
            self.visualizer.plot_current_losses(
                {"train_loss": loss.item(), "train_acc": 100 * (correct / total)},
                step=self.global_train_steps,
                individual=True,
            )

        mse = running_mse / batch_idx
        print("Epoch %d, loss = %.4f, batch_idx= %d" % (epoch, mse, batch_idx))
        print(
            "Epoch: %d Accuracy of the Train Images: %f"
            % (epoch, 100 * correct / total)
        )
        print("Confusion Matrix", confusion_matrix)
        train_accuracy = 100 * (correct / total)
        train_loss = mse

        train_classification_report = metrics.classification_report(
            y_true=np.concatenate(np.array(all_g)),
            y_pred=np.concatenate(np.array(all_p)),
            target_names=list(self.label_map.keys()),
            output_dict=True,
        )

        return train_loss, train_accuracy, train_classification_report

    def validate_epoch(self, epoch, best_val_acc_dict, best_val_loss_dict, inc_layer):
        self.model.eval()
        correct, total = 0, 0
        confusion_matrix = torch.zeros(3, 3)
        running_mse = 0
        batch_idx = 0

        all_p, all_g = [], []

        with torch.no_grad():
            for (inputs, labels) in tqdm(self.valid_loader):
                self.model.train()

                batch_idx += 1
                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)

                out = self.model(inputs)
                loss = self.criterion(out, labels.squeeze(1).long())
                running_mse += loss.item()
                # train_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(out.data, 1)

                total += labels.squeeze(1).size(0)
                correct += (predicted == labels.squeeze(1)).sum().item()

                all_p.append(predicted.detach().cpu().numpy())
                all_g.append(labels.squeeze(1).detach().cpu().numpy())

                for t, p in zip(labels.squeeze(1).view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                self.global_val_steps += 1
                self.visualizer.plot_current_losses(
                    {"val_loss": loss.item(), "val_acc": 100 * (correct / total)},
                    step=self.global_val_steps,
                    individual=True,
                )

        running_mse /= batch_idx
        if best_val_acc_dict[f"val_acc_{inc_layer}"] < 100 * (correct / total):
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.args.saved_models, f"best_val_acc_{inc_layer}_model.pt"
                ),
            )
            best_val_acc_dict[f"val_acc_{inc_layer}"] = 100 * (correct / total)
            # history["best_accuracy"] = best_accuracy

        if best_val_loss_dict[f"val_loss_{inc_layer}"] > running_mse:
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.args.saved_models, f"best_val_loss_{inc_layer}_model.pt"
                ),
            )
            best_val_acc_dict[f"val_loss_{inc_layer}"] = running_mse

        print(
            "Epoch: %d Accuracy of the Valid Images: %f"
            % (epoch, 100 * correct / total)
        )
        print("Confusion Matrix", confusion_matrix)
        valid_accuracy = 100 * (correct / total)
        valid_loss = running_mse

        self.scheduler.step(valid_loss)

        validation_classification_report = metrics.classification_report(
            y_true=np.concatenate(np.array(all_g)),
            y_pred=np.concatenate(np.array(all_p)),
            target_names=self.label_map.keys(),
            output_dict=True,
        )

        return (
            valid_loss,
            valid_accuracy,
            best_val_acc_dict,
            best_val_loss_dict,
            validation_classification_report,
        )

    def test_epoch(self, epoch, best_test_acc_dict, inc_layer):
        self.model.eval()
        correct, total, running_loss = 0, 0, 0.0
        all_p, all_g = [], []
        batch_idx = 0

        confusion_matrix = torch.zeros(3, 3)
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader):
                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)
                out = self.model(inputs)
                batch_idx += 1

                _, predicted = torch.max(out.data, 1)
                loss = self.criterion(out, labels.squeeze(1).long())
                running_loss += loss.item()
                # running_loss += loss.item() * inputs.size(0)

                total += labels.squeeze(1).size(0)
                correct += (predicted == labels.squeeze(1)).sum().item()

                all_p.append(predicted.detach().cpu().numpy())
                all_g.append(labels.squeeze(1).detach().cpu().numpy())

                for t, p in zip(labels.squeeze(1).view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                self.global_test_steps += 1
                self.visualizer.plot_current_losses(
                    {"test_loss": loss.item(), "test_acc": 100 * (correct / total)},
                    step=self.global_test_steps,
                    individual=True,
                )

        running_loss /= batch_idx
        if best_test_acc_dict[f"test_acc_{inc_layer}"] < 100 * (correct / total):
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.args.saved_models, f"best_test_acc_{inc_layer}_model.pt"
                ),
            )
            best_test_acc_dict[f"test_acc_{inc_layer}"] = 100 * (correct / total)

        print(
            fg(255, 10, 10)
            + "Accuracy of the Test Images: %f" % (100 * correct / total)
            + " best: "
            + str(best_test_acc_dict[f"test_acc_{inc_layer}"])
            + fg.rs
        )
        print("Confusion Matrix", confusion_matrix)

        test_accuracy = 100 * (correct / total)
        test_loss = running_loss
        test_classification_report = metrics.classification_report(
            y_true=np.concatenate(np.array(all_g)),
            y_pred=np.concatenate(np.array(all_p)),
            target_names=self.label_map.keys(),
            output_dict=True,
        )
        print(test_classification_report)

        return test_loss, test_accuracy, best_test_acc_dict, test_classification_report
