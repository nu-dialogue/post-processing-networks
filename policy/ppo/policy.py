import numpy as np
import torch
from convlab2.policy.ppo.multiwoz import PPOPolicy

from util import DEVICE
from policy import AbstractPolicy, PolicyOutput

class MyPPOPolicy(AbstractPolicy):
    def __init__(self, module_type, module_config) -> None:
        super().__init__(module_type, module_config)
        assert self.type == "policy"
        assert self.name == "ppo"
        self.ppo_policy = PPOPolicy()
    
    @property
    def module_state_dim(self) -> int:
        return self.ppo_policy.vector.da_dim

    def module_state_vector(self) -> np.ndarray:
        assert self.predicted_probas.shape[0] == self.module_state_dim
        return self.predicted_probas

    def init_session(self) -> None:
        self.ppo_policy.init_session()
        self.predicted_probas = np.zeros(self.ppo_policy.vector.da_dim)

    def predict(self, state: dict):
        s_vec = torch.Tensor(self.ppo_policy.vector.state_vectorize(state))
        
        with torch.no_grad():
            a = self.ppo_policy.policy.select_action(s_vec.to(device=DEVICE), False)
            a_weights = self.ppo_policy.policy.forward(s_vec.to(device=DEVICE))
        
        self.predicted_probas = torch.sigmoid(a_weights).cpu().detach().numpy()
        # predicted_action_vec = a.cpu().detach().numpy()
        predicted_action = self.ppo_policy.vector.action_devectorize(a)

        return PolicyOutput(system_action=predicted_action)# , predicted_action_vec
