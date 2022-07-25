import os
import pickle
import math
import json
import torch
from torch.utils.data import DataLoader

from util import get_logger
logger = get_logger(__name__)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, s_s, a_s):
        self.s_s = s_s
        self.a_s = a_s
        self.num_total = len(s_s)
    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        return s, a
    def __len__(self):
        return self.num_total


class BehaviorCloneDataset:
    def __init__(self, ppn_type) -> None:
        self.ppn_type = ppn_type

        self.observations = []
        self.target_actions = []

    def append(self, observation, target_action):
        self.observations.append(observation)
        self.target_actions.append(target_action)

    def extend(self, other_datasets: 'BehaviorCloneDataset'):
        self.observations += other_datasets.observations
        self.target_actions += other_datasets.target_actions

    def save(self, bcd_dpath):
        logger.info(f"Save {self.ppn_type}'s bc dataset to {bcd_dpath}")
        bcd_dpath = os.path.join(bcd_dpath, "data", self.ppn_type)
        if not os.path.exists(bcd_dpath):
            os.makedirs(bcd_dpath)
        pickle.dump(self.observations, open(os.path.join(bcd_dpath, "observations.pickle"), "wb"))
        pickle.dump(self.target_actions, open(os.path.join(bcd_dpath, "target_actions.pickle"), "wb"))

    def load(self, bc_update_module_combination, bcd_dpath):
        bcd_generate_config = json.load(open(os.path.join(bcd_dpath, "bcd_generate_config.json")))
        for module_type, module_name in bc_update_module_combination.items():
            if bcd_generate_config["module_combination"][module_type] != module_name:
                raise Exception("Module combination between 'bcd_generate ({})' and 'bc_update ({})' is inconsistent.".format(
                    bcd_generate_config["module_combination"], bc_update_module_combination
                ))
        bcd_dpath = os.path.join(bcd_dpath, "data", self.ppn_type)
        logger.info(f"Loading {self.ppn_type}'s bc dataset from {bcd_dpath}.")
        self.observations = pickle.load(open(os.path.join(bcd_dpath, "observations.pickle"), "rb"))
        self.target_actions = pickle.load(open(os.path.join(bcd_dpath, "target_actions.pickle"), "rb"))

    def make_dataloaders(self, total_size, train_size_ratio, batch_size, model_type):
        if len(self.observations) < total_size:
            raise Exception("The original dataset is insufficient. original({}) < total({})".format(len(self.observations), total_size))
        
        train_size = math.ceil(total_size * train_size_ratio)

        # When splitting into train and valid after sensor construction,
        # we get an error "ValueError: bad value(s) in fds_to_keep" when splitting the process.
        # https://discuss.pytorch.org/t/understanding-minimum-example-for-torch-multiprocessing/101010
        train_obs = torch.Tensor(self.observations[:total_size][:train_size])
        valid_obs = torch.Tensor(self.observations[:total_size][train_size:])
        if model_type == "multi_binary":
            dtype = torch.float32
        elif model_type == "discrete":
            dtype = torch.int64
        else:
            raise Exception("Unknown model type {} is detected on {}.".format(model_type, self.ppn_type))
        train_acts = torch.tensor(self.target_actions[:total_size][:train_size], dtype=dtype)
        valid_acts = torch.tensor(self.target_actions[:total_size][train_size:], dtype=dtype)

        train_dataloader = DataLoader(dataset=Dataset(train_obs, train_acts),
                                      batch_size=batch_size,
                                      shuffle=True)
        valid_dataloader = DataLoader(dataset=Dataset(valid_obs, valid_acts),
                                      batch_size=batch_size,
                                      shuffle=True)
        return train_dataloader, valid_dataloader