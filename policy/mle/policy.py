import json
import os
import torch
import numpy as np
from convlab2.policy.mle.multiwoz import MLEPolicy

from policy import AbstractPolicy, PolicyOutput
from util import ROOT_DPATH, DEVICE
MLE_DIRECTORY = os.path.join(ROOT_DPATH, "ConvLab-2/convlab2/policy/mle")

class MyMLEPolicy(AbstractPolicy):
    def __init__(self, module_type, module_config) -> None:
        super().__init__(module_type, module_config)
        assert self.type == "policy"
        assert self.name == "mle"
        model_fpath = os.path.join(MLE_DIRECTORY, "save", "best_mle.pol.mdl")
        self.mle_policy = MLEPolicy(model_file=model_fpath)
        self.mle_policy.policy.eval()
    
    @property
    def module_state_dim(self) -> int:
        return self.mle_policy.vector.da_dim

    def module_state_vector(self) -> np.ndarray:
        assert self.predicted_probas.shape[0] == self.module_state_dim
        return self.predicted_probas

    def init_session(self) -> None:
        self.mle_policy.init_session()
        self.predicted_probas = np.zeros(self.mle_policy.vector.da_dim)

    def predict(self, state: dict):
        s_vec = torch.Tensor(self.mle_policy.vector.state_vectorize(state))
        
        with torch.no_grad():
            a = self.mle_policy.policy.select_action(s_vec.to(device=DEVICE), False)
            a_weights = self.mle_policy.policy.forward(s_vec.to(device=DEVICE))
        
        self.predicted_probas = torch.sigmoid(a_weights).cpu().detach().numpy()
        # predicted_action_vec = a.cpu().detach().numpy()
        predicted_action = self.mle_policy.vector.action_devectorize(a)

        return PolicyOutput(system_action=predicted_action)# , predicted_action_vec
