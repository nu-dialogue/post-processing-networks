# Most of this implementation is based on 
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from ppn.distributions import Bernoulli, Categorical, DiagGaussian
from ppn.storage import RolloutStorage
from ppn.utils import init, update_linear_schedule
from util import DEVICE, get_logger
logger = get_logger(__name__)

class ActorCritic(nn.Module):
    def __init__(self, model_type, activation_type, obs_dim, hid_dim, act_dim, initialize_weight):
        super(ActorCritic, self).__init__()

        if initialize_weight:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        else:
            init_ = lambda m: m

        if activation_type == "relu":
            activaton_func = nn.ReLU
        elif activation_type == "tanh":
            activaton_func = nn.Tanh
        else:
            raise NotImplementedError("{} is unknown activation function type.".format(activation_type))
        
        self.actor = nn.Sequential(
            init_(nn.Linear(obs_dim, hid_dim)), activaton_func(),
            init_(nn.Linear(hid_dim, hid_dim)), activaton_func())

        self.critic = nn.Sequential(
            init_(nn.Linear(obs_dim, hid_dim)), activaton_func(),
            init_(nn.Linear(hid_dim, hid_dim)), activaton_func())

        self.critic_linear = init_(nn.Linear(hid_dim, 1))

        if model_type == "discrete":
            self.dist = Categorical(hid_dim, act_dim, initialize_weight)
        elif model_type == "continuous":
            # self.dist = DiagGaussian(hid_dim, act_dim, initialize_weight)
            raise NotImplementedError
        elif model_type == "multi_binary": 
            self.dist = Bernoulli(hid_dim, act_dim, initialize_weight)
        else:
            raise NotImplementedError("model type {} is not implemented.".format(model_type))

    def critic_forward(self, obs):
        hidden_critic = self.critic(obs)
        V = self.critic_linear(hidden_critic)
        return V

    def actor_forward(self, obs):
        hidden_actor = self.actor(obs)
        return hidden_actor

    def forward(self, obs):
        V = self.critic_forward(obs)
        hidden_actor = self.actor_forward(obs)
        return V, hidden_actor

    def get_action(self, obs, deterministic=False):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.unsqueeze(0) # [obs_dim] => [1, obs_dim]

        value, hidden_actor = self.forward(obs) # value: [1, 1], hidden_actor: [1, hid_dim]
        
        dist = self.dist(hidden_actor)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample() # [1, act_dim]

        log_probs = dist.log_probs(action) # [1, 1]
        # dist_entropy = dist.entropy().mean()

        value = value.squeeze(0) # [1, obs_dim] => [obs_dim]
        action = action.squeeze(0) # [1, act_dim] => [act_dim]
        log_probs = log_probs.squeeze(0) # [1, 1] => [1]

        return value.detach(), action.detach(), log_probs.detach()

    def get_value(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.unsqueeze(0) # [obs_dim] => [1, obs_dim]

        value = self.critic_forward(obs) # value: [1, 1]
        value = value.squeeze(0) # [1, obs_dim] => [obs_dim]

        return value.detach()

    def get_logits(self, obs):
        """
        Directly generates logits (values before being transformed
        into probability distributions) output by dist.linear
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.unsqueeze(0)
        value, hidden_actor = self.forward(obs)
        logits = self.dist(hidden_actor, as_logits=True)
        return logits.squeeze(0)

    def evaluate_actions(self, batch_obs, batch_acts):
        value, actor_hidden = self.forward(batch_obs) # [batch_size, obs_dim]
                                                   # => value: [batch_size, 1], hidden_actor: [batch_size, hid_dim]

        dist = self.dist(actor_hidden)
        log_probs = dist.log_probs(batch_acts) # [batch_size, 1]
        
        dist_entropy = dist.entropy().mean() # scalar

        value = value.squeeze(-1) # [batch_size, 1] => [batch_size]
        log_probs = log_probs.squeeze(-1) # [batch_size, 1] => [batch_size]

        return value, log_probs, dist_entropy

class PPO:
    def __init__(self, ppn_type, model_params):
        self.ppn_type = ppn_type
        self.model_type = model_params["model_type"]

        self.obs_dim = model_params["obs_dim"]
        self.hid_dim = model_params["hid_dim"]
        self.act_dim = model_params["act_dim"]
        self.actor_critic = ActorCritic(model_type=model_params["model_type"],
                                        activation_type=model_params["activation_type"],
                                        obs_dim=self.obs_dim,
                                        hid_dim=self.hid_dim,
                                        act_dim=self.act_dim,
                                        initialize_weight=model_params["initialize_weight"])

        self.deterministic_train = model_params["deterministic_train"]
        self.deterministic_action: bool

        self.clip_param = model_params["clip_param"]
        self.epoch_num = model_params["epoch_num"]
        self.mini_batch_size = model_params["mini_batch_size"]

        self.value_loss_coef = model_params["value_loss_coef"]
        self.entropy_coef = model_params["entropy_coef"]

        self.max_grad_norm = model_params["max_grad_norm"]
        self.use_clipped_value_loss = model_params["use_clipped_value_loss"]

        self.use_linear_lr_decay = model_params["use_linear_lr_decay"]
        self.initial_lr = model_params["lr"]
        if model_params["optimizer"] == "adam":
            self.optimizer = Adam(self.actor_critic.parameters(), lr=model_params["lr"], eps=model_params["eps"])
        elif model_params["optimizer"] == "rmsprop":
            self.optimizer = RMSprop(self.actor_critic.parameters(), lr=model_params["lr"], eps=model_params["eps"])

    def train_iteration(self):
        self.deterministic_action = self.deterministic_train
        self.actor_critic.to("cpu")
        self.actor_critic.eval()

    def test_iteration(self):
        self.deterministic_action = True
        self.actor_critic.to("cpu")
        self.actor_critic.eval()

    def get_action(self, observation):
        with torch.no_grad():
            value, action, log_probs = self.actor_critic.get_action(obs=observation,
                                                                    deterministic=self.deterministic_action)
        return value, action, log_probs

    def update(self, rollouts: RolloutStorage, sampled_timesteps: int, total_timesteps: int):
        if self.use_linear_lr_decay:
            update_linear_schedule(self.optimizer, sampled_timesteps, total_timesteps, self.initial_lr)

        self.actor_critic.to(DEVICE)
        self.actor_critic.train()

        rollouts.to(DEVICE)
        rollouts.train()

        logger.info("Updating {}'s model...".format(self.ppn_type))
        total_actor_loss = torch.zeros(self.epoch_num)
        total_critic_loss = torch.zeros(self.epoch_num)
        for i in range(self.epoch_num):
            mini_batch_generator = rollouts.get_mini_batch_generator()
            for j, mini_batch in enumerate(mini_batch_generator):
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs, dist_entropy = self.actor_critic.evaluate_actions(mini_batch.obs, mini_batch.acts)

                # Calculate surrogate losses.
                ratios = torch.exp(curr_log_probs - mini_batch.log_probs)
                surr1 = ratios * mini_batch.advs
                surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * mini_batch.advs
                actor_loss = (-torch.min(surr1, surr2)).mean()
                
                # Calculate value losses
                if self.use_clipped_value_loss:
                    value_losses = (V - mini_batch.rtgs).pow(2)
                    value_pred_clipped = mini_batch.values + (V - mini_batch.values).clamp(-self.clip_param, self.clip_param)
                    value_losses_clipped = (value_pred_clipped - mini_batch.rtgs).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (V - mini_batch.rtgs).pow(2).mean()
                    # value_loss = F.mse_loss(V, mini_batch.rtgs)
                critic_loss = value_loss

                # Backward & Step
                self.optimizer.zero_grad()
                (critic_loss * self.value_loss_coef + actor_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_actor_loss[i] += actor_loss.detach().cpu()
                total_critic_loss[i] += critic_loss.detach().cpu()
            total_actor_loss[i] /= j+1
            total_critic_loss[i] /= j+1
        logger.info("actor loss: {:.5f}\tvalue loss: {:.5f}".format(total_actor_loss.mean(), total_critic_loss.mean()))
        return {"actor_losses": total_actor_loss, "critic_losses": total_critic_loss}

    def save_model(self, model_fpath):
        torch.save(self.actor_critic.state_dict(), model_fpath)

    def load_model(self, model_fpath):
        self.actor_critic.load_state_dict(torch.load(model_fpath, map_location=DEVICE))
