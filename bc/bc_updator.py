import torch
import torch.nn as nn
from util import DEVICE, get_logger
logger = get_logger(__name__)
from ppn.ppo import PPO
from bc.utils import comupte_F1, comupte_accuracy
from bc.bc_dataset import BehaviorCloneDataset

class BehaviorCloneUpdator:
    def __init__(self, ppn_type, bc_config, ppo: PPO):
        self.ppn_type = ppn_type
        self.model_type = ppo.model_type
        self.bc_update_module_combination = bc_config["module_combination"]
        self.bc_dataset_dpath = bc_config["bc_dataset_dpath"]
        self.bc_dataset_total_size = bc_config["bc_dataset_total_size"]
        self.bc_dataset_train_size_ratio = bc_config["bc_dataset_train_size_ratio"]
        self.epoch_num = bc_config["epoch_num"]
        self.mini_batch_size = bc_config["mini_batch_size"]
        self.early_stopping_patience = bc_config["early_stopping_patience"]

        self.actor_critic =  ppo.actor_critic
        self.obs_dim = ppo.obs_dim
        self._set_criterion()
        self._set_optimizer(bc_config)
        self._set_bc_dataloaders()

    def _set_criterion(self):
        if self.model_type == "multi_binary":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.model_type == "discrete":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def _set_optimizer(self, bc_config):
        if bc_config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=bc_config["learning_rate"])
        elif bc_config["optimizer"] == "rmsprop":
            self.optimizer = torch.optim.RMSProp(self.actor_critic.paramters(), lr=bc_config["learning_rate"])
        else:
            raise NotImplementedError

    def _set_bc_dataloaders(self):
        bc_dataset = BehaviorCloneDataset(ppn_type=self.ppn_type)
        bc_dataset.load(bc_update_module_combination=self.bc_update_module_combination,
                        bcd_dpath=self.bc_dataset_dpath)
        self.train_dataloader, self.valid_dataloader = \
            bc_dataset.make_dataloaders(total_size=self.bc_dataset_total_size,
                                        train_size_ratio=self.bc_dataset_train_size_ratio,
                                        batch_size=self.mini_batch_size,
                                        model_type=self.model_type)
        self.train_batch_num = len(self.train_dataloader)
        self.valid_batch_num = len(self.valid_dataloader)

    def train_epoch(self, epoch_id):
        losses = []
        self.actor_critic.train()
        self.actor_critic.to(DEVICE)
        for input_obs, target_acts in self.train_dataloader:
            input_obs = input_obs[:, :self.obs_dim].to(DEVICE)
            target_acts = target_acts.to(DEVICE)
            self.optimizer.zero_grad()
            act_weights = self.actor_critic.get_logits(input_obs)
            try:
                loss = self.criterion(act_weights,target_acts)
            except Exception:
                logger.info("act_weights.shape", act_weights.shape)
                logger.info("target_acts.shape", target_acts.shape)
                raise
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        train_loss = sum(losses)/len(losses)
        logger.info("<BC TRAIN: {}> Epoch: {:>3}\tLoss: {:.8f}".format(self.ppn_type, epoch_id, train_loss))

    def valid_epoch(self, epoch_id):
        losses = []
        score_history = []
        self.actor_critic.eval()
        self.actor_critic.to(DEVICE)
        for input_obs, target_acts in self.valid_dataloader:
            input_obs = input_obs[:, :self.obs_dim].to(DEVICE)
            target_acts = target_acts.to(DEVICE)
            act_weights = self.actor_critic.get_logits(input_obs)
            loss = self.criterion(act_weights, target_acts)
            losses.append(loss.item())
            score = self.evaluate_step(act_weights=act_weights, target_acts=target_acts)
            score_history.append(score)
        val_loss = sum(losses)/len(losses)
        total_score = self.evaluate_total(score_history=score_history)
        log_str = "<BC VALID: {}> Epoch: {:>3}\tLoss: {:.8f}\t".format(self.ppn_type, epoch_id, val_loss)
        log_str += "\t".join(["{}: {:.5f}".format(key, value) for key, value in total_score.items()])
        logger.info(log_str)
        
        if self.__best_val_loss is None or self.__best_val_loss > val_loss:
            self.__count  = 0
            self.__best_val_loss = val_loss
            self.__best_actor_sd = self.actor_critic.state_dict() # no need deepcopy
        else:
            self.__count += 1
        early_stop = False
        if self.early_stopping_patience and self.__count >= self.early_stopping_patience:
            early_stop = True
        return early_stop

    def evaluate_step(self, act_weights, target_acts):
        if self.model_type == "multi_binary":
            return comupte_F1(act_weights, target_acts, score_history=None)
        elif self.model_type == "discrete":
            return comupte_accuracy(act_weights, target_acts, score_history=None)
    
    def evaluate_total(self, score_history):
        if self.model_type == "multi_binary":
            return comupte_F1(act_weights=None, target_acts=None, score_history=score_history)
        elif self.model_type == "discrete":
            return comupte_accuracy(act_weights=None, target_acts=None, score_history=score_history)
    
    def update(self):
        self.__count = 0
        self.__best_val_loss = None
        self.__best_actor_sd = None

        for epoch_id in range(1, self.epoch_num+1):
            self.train_epoch(epoch_id)
            early_stop = self.valid_epoch(epoch_id)
            if early_stop:
                logger.info("Early Stopped.")
                break
        self.actor_critic.load_state_dict(self.__best_actor_sd)