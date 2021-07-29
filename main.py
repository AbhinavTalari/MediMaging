import torch
import time
from dataloader import get_dataloader
import argparse
import numpy as np
from torchvision.utils import save_image
import os
from trainer import Trainer
import pandas as pd
import logging
import sys
from configs import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Issues: 1) scheduler
        2) LR Decay

1) Need to check the codes. Issues with the val accc --> done 
2) Testing needs to be performed only when val auc is high --> less important
3) Need to save test results too (done)
4) Save confusion matrix, f1, p and r. (done)
5) Image normalization transforms.Normalize(means,std) -> (leave for now)
6) Need to add schedulers(missing the code) --> one liner in val-epoch (done)
7) weighted cross entropy --> as an option (include when training done on full dataset)
8) Done with Tensorboard. Check the code
9) Write loggers to a file --> less important(leave this one)
10) Need to update image sizes (use dict to hardcode this part)
    (or) config file for this one (done)
11) resolved few bugs
12) Loading Pretrained weights function needs to be modified --> less imp
13) add weight init for resnet i.e xavier or he init (done)
14) pick augmentation values based on model size
15) freeze previous layers when new layer is added (done)
"""
if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser(description="Curriculum Learning")

        parser.add_argument(
            "--dataset_name",
            type=str,
            default="disease_data",
            help="disease_data",
        )
        parser.add_argument(
            "--saved_models",
            type=str,
            default="./models",
            help="Saved models directory",
        )
        parser.add_argument(
            "--resnet_variant",
            type=int,
            default=18,
            help="18, 34, 50, 101",
        )
        parser.add_argument(
            "--load_pretrained",
            type=bool,
            default=True,
            help="load pre trained weights for model",
        )
        parser.add_argument("--num_out_classes", type=int, default=3)
        parser.add_argument("--min_resnet_blocks", type=int, default=1, help="1,2,3,4")
        parser.add_argument(
            "--max_resnet_blocks",
            type=int,
            default=3,
            help="1,2,3,4 and >= min_resnet_blocks",
        )
        parser.add_argument("--base_dir", type=str, default=os.getcwd())
        parser.add_argument("--train_batch_size", type=int, default=16)
        parser.add_argument("--test_batch_size", type=int, default=16)

        parser.add_argument("--total_train_images", type=int, default=1000)
        parser.add_argument("--total_test_images", type=int, default=500)
        parser.add_argument("--nThreads", type=int, default=8)
        parser.add_argument("--max_epochs", type=int, default=50)
        # parser.add_argument("--eval_freq_iter", type=int, default=200)
        # parser.add_argument("--print_freq_iter", type=int, default=50)
        parser.add_argument("--splitTrain", type=float, default=0.7)
        parser.add_argument("--validation_split", type=float, default=0.2)
        parser.add_argument(
            "--training", type=str, default="edge", help="sketch / rgb / edge"
        )
        parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
        parser.add_argument(
            "--scratch",
            type=bool,
            default=False,
            help="enable if training from scratch",
        )

        # ----------------------------------optimizer-------------------------------------
        parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="learning rate for matching net trainer",
        )
        parser.add_argument(
            "--optim",
            type=str,
            default="adam",
            help="optimizer for matchnum_blocksing net trainer",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.0,
            help="weights decay for matching net trainer",
        )

        parser.add_argument(
            "--scheduler",
            type=str,
            help="The learning rate scheduler, step_lr or cyclic_lr",
        )

        parser.add_argument(
            "--resize_shape",
            type=int,
            default=256,
        )

        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--freeze_prev_layers", type=bool, default=False)

        params = parser.parse_args()
        print(vars(params))
        # logger.info(f"The args are: {vars(params)}")

        os.system(f"mkdir -p {params.saved_models}")

        # load model and data in steps
        num_blocks_to_img_size_dict = config.num_blocks_to_img_size_dict

        for inc_layer in range(params.min_resnet_blocks, params.max_resnet_blocks + 1):
            # initialize the dataloader
            img_size = num_blocks_to_img_size_dict.get(str(inc_layer), 64 * 4)
            dataloaders_dict = get_dataloader(params, img_size)

            logger.info(
                f"The dataloaders are loaded successfully with image_width = {img_size}"
            )

            # initialize the trainer object
            trainer_obj = Trainer(params, dataloaders_dict, inc_layer)

            logger.info(f"Trainer loaded successully with the dataloaders!")

            # intialize variables
            epochs = []
            train_loss, train_acc, train_classification_reports = [], [], []
            val_loss, val_acc, val_classification_reports = [], [], []
            test_loss, test_acc, test_classification_reports = [], [], []

            best_val_acc_dict = dict()
            best_val_loss_dict = dict()
            best_test_acc_dict = dict()

            best_val_acc_dict[f"val_acc_{inc_layer}"] = 0
            best_val_loss_dict[f"val_loss_{inc_layer}"] = sys.maxsize
            best_test_acc_dict[f"test_acc_{inc_layer}"] = 0

            for e in range(params.max_epochs):
                (
                    total_train_loss,
                    total_train_acc,
                    train_classification_report,
                ) = trainer_obj.train_epoch(e)

                with torch.no_grad():
                    (
                        total_val_loss,
                        total_val_acc,
                        best_val_acc_dict,
                        best_val_loss_dict,
                        val_classification_report,
                    ) = trainer_obj.validate_epoch(
                        e, best_val_acc_dict, best_val_loss_dict, inc_layer
                    )

                # print("---------###############----------")
                logger.debug(
                    "\nEpoch {}: train_loss:{} train_accuracy:{}".format(
                        e, total_train_loss, total_train_acc
                    )
                )
                logger.debug(
                    "Epoch {}: val_loss:{} val_accuracy:{}\n".format(
                        e, total_val_loss, total_val_acc
                    )
                )
                # print("---------###############----------")

                with torch.no_grad():
                    (
                        total_test_loss,
                        total_test_acc,
                        best_test_acc_dict,
                        test_classification_report,
                    ) = trainer_obj.test_epoch(e, best_test_acc_dict, inc_layer)

                """
                # @TODO need to setup val/test sets
                if total_val_acc > best_val_acc:
                    best_val_acc = total_val_acc
                    total_test_loss, total_test_accuracy = trainer_obj.test_epoch()
                    print("Epoch {}: test_loss:{} test_accuracy:{}".format(e, total_test_loss, total_test_accuracy))
                """
                epochs.append(e)
                train_loss.append(total_train_loss)
                train_acc.append(total_train_acc)
                train_classification_reports.append(train_classification_report)
                val_loss.append(total_val_loss)
                val_acc.append(total_val_acc)
                val_classification_reports.append(val_classification_report)
                test_loss.append(total_test_loss)
                test_acc.append(total_test_acc)
                test_classification_reports.append(test_classification_report)

                columns = [
                    "Epochs",
                    "Train Accuracy",
                    "Train Loss",
                    "Train classification report",
                    "Val Accuracy",
                    "Val Loss",
                    "Val Classification Report",
                    "Test Loss",
                    "Test Acc",
                    "Test Classification Report",
                ]

                stats_df = pd.DataFrame(
                    list(
                        zip(
                            epochs,
                            train_acc,
                            train_loss,
                            train_classification_reports,
                            val_acc,
                            val_loss,
                            val_classification_reports,
                            test_loss,
                            test_acc,
                            test_classification_reports,
                        )
                    ),
                    columns=columns,
                )

                results_fname = os.path.join(
                    params.saved_models, f"results_{inc_layer}.csv"
                )
                stats_df.to_csv(results_fname, header=columns, index=False)

    except Exception as exc:
        logger.error(f"{exc}")
