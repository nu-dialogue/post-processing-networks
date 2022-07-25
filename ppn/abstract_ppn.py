from abc import ABC, abstractmethod
import os
import json
import numpy as np

from ppn.ppo import PPO
from ppn.storage import RolloutStorage
from bc import BehaviorCloneDataset, BehaviorCloneUpdator
from util import AbstractModule, ROOT_DPATH, get_logger
logger = get_logger(__name__)
OUTPUTS_DPATH = os.path.join(ROOT_DPATH, "outputs")

class AbstractPPN(ABC):
    def __init__(self, module: AbstractModule, system_state_dim: int):
        self.module = module
        self.module_name = module.name
        self.type = "{}_ppn".format(module.type)
        
        # 1. Set basic settings
        ppn_config = module.ppn_config
        assert ppn_config["ppn_type"] == self.type
        self.run_id = ppn_config["run_id"] # run_id
        self.use = ppn_config["use"] # bool
        self.rl_train = ppn_config["rl_train"] # bool
        
        # 2-1. Set the model output path and config save path for this time
        self.trained_model_dpath = ppn_config["trained_model_dpath"]
        self.ppn_config_fpath = ppn_config["ppn_config_fpath"]
        # 2-2. Set the path of the pretrained model to be used
        self.resume_rl_train_id = ppn_config["resume_rl_train_id"]
        self.resume_rl_iteration_id = ppn_config["resume_rl_iteration_id"]
        self.resume_rl_trained_model_fpath = ppn_config["resume_rl_trained_model_fpath"]

        # 3. Set up the vocabulary of target module
        self._prepare_vocab()

        # 4. Prepare other options for PPN
        self.use_system_state = ppn_config["use_system_state"]
        self.ignore_new_domain = False # ppn_config["ignore_new_domain"]

        # 5. Set MLP parameters based on the set up vocabulary
        self.system_state_dim = system_state_dim if self.use_system_state else 0
        self.model_params = ppn_config["model_params"]
        self._set_model_params()
        if None in list(self.model_params.values()):
            logger.info(json.dumps(self.model_params, indent=4))
            raise Exception("{}:: Model hyper patameters are not set properly.".format(self.type))

        # 6. Instantiate a ppo manager
        self.ppo = PPO(ppn_type=self.type, model_params=self.model_params)
        if self.resume_rl_trained_model_fpath:
            logger.info("{}:: Loading rl trained model...".format(self.type))
            self.load_model( self.resume_rl_trained_model_fpath )
        else:
            logger.info("{}:: Pretrained model is not resumed ".format(self.type))

        # 7. Set Options for Behavior clone
        self.bc_config = ppn_config["bc_config"]

        # 8. Save configs
        self._save_config()

    @abstractmethod
    def _prepare_vocab(self):
        """
        Set vocabulary, etc. required for postprocess using instances of each module
        """
        pass

    @abstractmethod
    def _set_model_params(self):
        """
        Set MLP parameters based on the prepared vocabulary
        Basically set up obs_dim and act_dim
        """
        pass

    def bcd_generate(self):
        self.current_mode = "bcd_generate"
        self.bc_dataset = BehaviorCloneDataset(ppn_type=self.type)

    def prepare_bc_updator(self):
        self.bc_updator = BehaviorCloneUpdator(ppn_type=self.type, bc_config=self.bc_config, ppo=self.ppo)

    def train_iteration(self):
        """Execute before starting each iteration"""
        self.current_mode = "rl_train"
        self.ppo.train_iteration()
        self.rollouts = RolloutStorage(ppn_type=self.type, model_params=self.model_params)

    def test_iteration(self):
        self.current_mode = "test"
        self.ppo.test_iteration()

    def sample_episode(self):
        """Execute before starting each episode (dialogue)"""
        if self.current_mode == "test":
            pass
        elif self.current_mode == "rl_train":
            self.rollouts.sample_episode()
        elif self.current_mode == "bcd_generate":
            pass

    def _get_action(self, observation, final):
        """Execute every turn"""
        value, action, log_prob = self.ppo.get_action(observation=observation)
        if self.current_mode == "rl_train":
            if final:
                self.rollouts.end_episode(value)
            else:
                self.rollouts.insert_current(observation=observation, action=action, value=value, log_prob=log_prob)
        elif self.current_mode == "test":
            pass
        else:
            raise Exception("get action method can not be called on {} mode.".format(self.current_mode))
        return action

    def _store_observation_action(self, observation, target_action):
        self.bc_dataset.append(observation=observation, target_action=target_action)

    def save_reward(self, reward, done):
        """Execute every turn"""
        if self.current_mode == "rl_train":
            self.rollouts.insert_previous(reward=reward, done=done)
        elif self.current_mode in ["test", "bcd_generate"]:
            pass
        else:
            raise Exception("Unnknown mode {} is detected.".format(self.current_mode))
        return reward

    def get_ppo_rollouts(self):
        return self.rollouts.get_batch()

    def get_bc_dataset(self):
        return self.bc_dataset

    def merge_ppo_rollouts(self, ppo_rollouts):
        self.rollouts.merge(other_rollouts=ppo_rollouts)

    def merge_bc_dataset(self, bc_dataset):
        self.bc_dataset.extend(other_datasets=bc_dataset)

    @abstractmethod
    def _vectorize(self, module_output):
        pass

    @abstractmethod
    def _devectorize(self, ppn_output_vec, **items):
        pass

    @abstractmethod
    def _through_module_output(self, module_output):
        pass

    def postprocess(self, module_output, system_state, final):
        module_output_vec, items = self._vectorize(module_output)
        if self.use_system_state:
            observation = np.r_[module_output_vec, system_state]
        else:
            observation = np.r_[module_output_vec]
        assert observation.shape[0] == self.model_params["obs_dim"]

        if self.current_mode in ["rl_train", "test"]:
            ppn_output_vec = self._get_action(observation=observation, final=final)
            ppn_output = self._devectorize(ppn_output_vec, **items)
            return ppn_output
        elif self.current_mode == "bcd_generate":
            self._store_observation_action(observation=observation, target_action=module_output_vec)
            module_output = self._through_module_output(module_output)
            return module_output
        else:
            raise Exception("Unknown mode {} is detected.".format(self.current_mode))

    def ppo_update(self, sampled_timesteps, total_timesteps):
        """Execute after each iteration"""
        losses = self.ppo.update(rollouts=self.rollouts,
                                 sampled_timesteps=sampled_timesteps,
                                 total_timesteps=total_timesteps)
        return losses

    def bc_update(self):
        """behavior cloning"""
        self.bc_updator.update()

    def save_model(self, i):
        if not os.path.exists( self.trained_model_dpath ):
            os.makedirs( self.trained_model_dpath)
        model_fpath = os.path.join(self.trained_model_dpath, "{}.model".format(i))
        self.ppo.save_model(model_fpath)
        logger.info("{}:: Saved model to {}.".format(self.type, model_fpath))

    def load_model(self, model_fpath):
        self.ppo.load_model(model_fpath)
        logger.info("{}:: Loaded model from {}.".format(self.type, model_fpath))

    def _save_config(self):
        """Called from module PPN"""
        if os.path.exists(self.ppn_config_fpath):
            return
        attributes = self.__dict__.copy()
        for key, value in self.__dict__.items():
            # Delete attributes that meet the following criteria
            if key.startswith("_{}__".format(self.type)):
                # Variables starting with __
                del attributes[key]
            elif (not isinstance(value, (str, int, float, list, dict))):
                # Other than str, int, float, list, dict
                del attributes[key]
                
        if not os.path.exists( os.path.dirname(self.ppn_config_fpath)):
            os.makedirs(os.path.dirname(self.ppn_config_fpath))
        json.dump(attributes, open( self.ppn_config_fpath, "w"), indent=4)
