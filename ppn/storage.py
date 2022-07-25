from re import S
import torch
from collections import namedtuple
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

Batch = namedtuple("Batch", ["obs", "acts", "values", "log_probs", "rtgs", "advs"])

class RolloutStorage:
    def __init__(self, ppn_type, model_params):
        self.ppn_type = ppn_type
        self.mini_batch_size = model_params["mini_batch_size"]
        self.gamma = model_params["gamma"]
        self.gae_lambda = model_params["gae_lambda"]

        self.batch_obs = []
        self.batch_acts = []
        self.batch_values = []
        self.batch_log_probs = []

        self.batch_next_values = []
        self.batch_rews = []
        self.batch_dones = []

        self.batch_rtgs = []
        self.batch_advs = []

        self.batch_lens = []

        self.initial_merge = True

    def sample_episode(self):
        self.ep_values = []
        self.ep_rews = []
        self.ep_dones = []
        self.ep_t = 0

    def insert_current(self, observation, action, value, log_prob):
        self.batch_obs.append(observation) # [array([obs_dim]), ...]
        self.batch_acts.append(action) # [tensor([act_dim]), ...]
        self.batch_log_probs.append(log_prob) # [tensor([1]), ...]
        
        # self.batch_values.append(value) # [tensor([1]), ...]
        self.ep_values.append(value) # [tensor([1]), ...]

    def insert_previous(self, reward, done):
        self.ep_rews.append(reward)
        self.ep_dones.append(done)
        self.ep_t += 1
        
    def end_episode(self, next_value):
        self.batch_lens.append(self.ep_t)
        self.batch_next_values.append(next_value)
        self.batch_values.append(torch.tensor(self.ep_values)) # [tensor([timestep_num]), ... ]
        self.batch_rews.append(torch.tensor(self.ep_rews)) # [tensor([timestep_num]), ... ]
        self.batch_dones.append(self.ep_dones)
        # ep_rtgs, ep_advs = self.compute_rtgs_and_advs(...) # tensor([timestep_num]), tensor([timestep_num])
        # self.batch_rtgs.append(ep_rtgs)
        # self.batch_advs.append(ep_advs)

    def compute_rtgs_and_advs(self, ep_rews, ep_values, ep_dones, next_value):
        step_num = ep_rews.size(0)
        ep_advs = torch.zeros(step_num+1)
        last_value = next_value
        for step_id in reversed(range(step_num)):
            mask = 1.0 - ep_dones[step_id]
            delta = ep_rews[step_id] + self.gamma * last_value * mask - ep_values[step_id]
            ep_advs[step_id] = delta + self.gamma * self.gae_lambda * ep_advs[step_id+1] * mask
            last_value = ep_values[step_id]
        return ep_advs[:-1]+ep_values, ep_advs[:-1]

    def compute_rtgs_and_advs_(self):
        ep_num = len(self.batch_rews)

        for ep_id in range(ep_num):
            ep_rews = self.batch_rews[ep_id]
            ep_values = self.batch_values[ep_id]
            ep_dones = self.batch_dones[ep_id]

            step_num = ep_rews.size(0)
            ep_advs = torch.zeros(step_num+1)
            last_value = self.batch_next_values[ep_id]

            for step_id in reversed(range(step_num)):
                mask = 1.0 - ep_dones[step_id]
                delta = ep_rews[step_id] + self.gamma * last_value * mask - ep_values[step_id]
                ep_advs[step_id] = delta + self.gamma * self.gae_lambda * ep_advs[step_id+1] * mask
                last_value = ep_values[step_id]
                
            self.batch_advs.append(ep_advs[:-1])
            self.batch_rtgs.append(ep_advs[:-1] + ep_values)

    def to(self, device):
        self.batch_rtgs = self.batch_rtgs.to(device)
        self.batch_advs = self.batch_advs.to(device)
        self.batch_values = self.batch_values.to(device)
        self.batch_obs = self.batch_obs.to(device)
        self.batch_acts = self.batch_acts.to(device)
        self.batch_log_probs = self.batch_log_probs.to(device)

    def get_batch(self):
        self.compute_rtgs_and_advs_()
        self.batch_rtgs = torch.cat(self.batch_rtgs).to(dtype=torch.float) # tensor([batch_size])
        self.batch_advs = torch.cat(self.batch_advs).to(dtype=torch.float) # tensor([batch_size])

        self.batch_values = torch.cat(self.batch_values).to(dtype=torch.float) # [tensor([timesteps]), ...] => tensor([batch_size])
        self.batch_obs = torch.tensor(self.batch_obs, dtype=torch.float) # [array([obs_dim], ...] => tensor([batch_size, obs_dim])
        self.batch_acts = torch.stack(self.batch_acts).to(dtype=torch.float)  # [tensor([act_dim], ...)] => tensor([batch_size, act_dim])
        self.batch_log_probs = torch.stack(self.batch_log_probs).to(dtype=torch.float).squeeze(-1) # [tensor([1]), ...] => tensor([batch_size])
        return self

    def merge(self, other_rollouts: 'RolloutStorage'):
        if self.initial_merge:
            self.initial_merge = False
            self.batch_rtgs = torch.tensor([])
            self.batch_advs = torch.tensor([])
            self.batch_values = torch.tensor([])
            self.batch_obs = torch.tensor([])
            self.batch_acts = torch.tensor([])
            self.batch_log_probs = torch.tensor([])

        self.batch_lens += other_rollouts.batch_lens
        self.batch_rtgs = torch.cat([self.batch_rtgs, other_rollouts.batch_rtgs])
        self.batch_advs = torch.cat([self.batch_advs, other_rollouts.batch_advs])
        self.batch_values = torch.cat([self.batch_values, other_rollouts.batch_values])
        self.batch_obs = torch.cat([self.batch_obs, other_rollouts.batch_obs])
        self.batch_acts = torch.cat([self.batch_acts, other_rollouts.batch_acts])
        self.batch_log_probs = torch.cat([self.batch_log_probs, other_rollouts.batch_log_probs])

    def train(self):
        current_batch_size = sum(self.batch_lens)
        assert current_batch_size == self.batch_obs.shape[0]
        assert current_batch_size == self.batch_acts.shape[0]
        assert current_batch_size == self.batch_log_probs.shape[0]
        assert current_batch_size == self.batch_rtgs.shape[0]
        assert current_batch_size == self.batch_advs.shape[0]
        assert current_batch_size == self.batch_values.shape[0]

        self.batch_advs = (self.batch_advs - self.batch_advs.mean()) / (self.batch_advs.std() + 1e-5) # normalization

    def get_mini_batch_generator(self):
        batch_size = self.batch_acts.shape[0]
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               self.mini_batch_size,
                               drop_last=True)

        for indices in sampler:
            yield Batch(self.batch_obs[indices], self.batch_acts[indices],
                        self.batch_values[indices], self.batch_log_probs[indices],
                        self.batch_rtgs[indices], self.batch_advs[indices])

    def end_single_iteration(self):
        raise NotImplementedError