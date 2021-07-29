import torch
import random
import torchvision.transforms as transforms
from configs import config

"""
!pip install gdown
!gdown --id 1fpoHMKGH-2Jv4AdxtlVPvWR9s6WEQhbi
!unzip  ./disease_data.zip

"""


def load_pretrained(
    model, pretrained_model, ignore_layers=[], verbose=False, state_dict=True
):

    model_dict = model.state_dict()
    if state_dict:
        pretrained_dict = pretrained_model
    else:
        pretrained_dict = pretrained_model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    for layer in ignore_layers:
        del pretrained_dict[layer]
        if verbose:
            print("{} weights are not updated".format(layer))
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("model loaded with pretrained weights...")

    return model


def _get_transformations(epoch):
    transformation_indexes = config.transformation_indexes
    transforms_list = []
    if 1 in transformation_indexes:
        if epoch in range(1, 10):
            max_ = 0.1
        elif epoch in range(10, 20):
            max_ = 0.2
        else:
            max_ = 0.3
        """
        brightness = random.uniform(0, max_)
        contrast = random.uniform(0, max_)
        saturation = random.uniform(0, max_)
        hue = random.uniform(0, max_)
        """
        brightness = max_
        contrast = max_
        saturation = max_
        hue = max_
        transforms_list.append(
            transforms.ColorJitter(brightness, contrast, saturation, hue)
        )
    if 2 in transformation_indexes:
        if epoch in range(1, 10):
            # degree = random.randint(1, 10)
            degree = 10
        elif epoch in range(10, 20):
            # degree = random.randint(10, 20)
            degree = 20
        else:
            # degree = random.randint(20, 30)
            degree = 30
        transforms_list.append(transforms.RandomAffine(degree))

    if 3 in transformation_indexes:
        transforms_list.append(transforms.RandomHorizontalFlip(p=1))

    # if 4 in transformation_indexes:
    # @ TODO
    #    transforms_list.append(transforms.RandomPerspective(p=1))

    if 5 in transformation_indexes:
        if epoch in range(1, 10):
            kernel_size = 3
            sigma = (0.1, 0.7)
        elif epoch in range(10, 20):
            kernel_size = 3
            sigma = (0.7, 1.3)
        else:
            kernel_size = 5
            sigma = (1.3, 2.0)

        transforms_list.append(transforms.GaussianBlur(kernel_size, sigma))

    final_transforms = transforms.Compose(transforms_list)
    return final_transforms


def get_transformations(epoch, phase="train"):
    if phase == "train":
        return _get_transformations(epoch)
    else:
        return None


# from resnet import ResidualNet
# block1 = ResidualNet('CIFAR100', 18, 3, 'CBAM', 1)
# block2 = ResidualNet('CIFAR100', 18, 3, 'CBAM', 2)

# block1 = torch.load('block1.pt')
# block2 = load_pretrained(block2, block1, ignore_layers=['fc.weight', 'fc.bias'], verbose=False)
