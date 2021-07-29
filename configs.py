from easydict import EasyDict as edict

config = edict()

config.base_size = 64
config.num_blocks_to_img_size_dict = {
    "1": config.base_size,
    "2": config.base_size * 3,
    "3": config.base_size * 6,
    "4": config.base_size * 8,
}

config.transformation_indexes = list(range(1, 7))
config.steps = [0, 10, 20]
