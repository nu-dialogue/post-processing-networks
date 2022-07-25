import numpy as np
from policy import PolicyOutput, PolicyPPNOutput
from ppn import AbstractPPN
from policy.ppo.policy import MyPPOPolicy

class MyPPOPolicyPPN(AbstractPPN):
    def __init__(self, module: MyPPOPolicy, system_state_dim: int) -> None:
        super().__init__(module=module, system_state_dim=system_state_dim)

    def _prepare_vocab(self):
        self.__vector = self.module.ppo_policy.vector
    
    def _set_model_params(self):
        self.model_params.update({
            "obs_dim": len(self.__vector.da_voc) + self.system_state_dim,
            "act_dim": len(self.__vector.da_voc)
        })

    def _vectorize(self, module_output: PolicyOutput):
        system_action = module_output.system_action
        da_vec = self.__vector.action_vectorize(system_action)
        domain_memory = set([domain for _intent, domain, _slot, _value in system_action])
        return da_vec, {"domain_memory": domain_memory}

    def _devectorize(self, processed_module_output_vec, **items):
        domain_memory = items["domain_memory"]
        dialog_acts = self.__vector.action_devectorize(processed_module_output_vec)
        processed_da = [f"{d}-{i}-{s}-{v}" for i,d,s,v in dialog_acts]
        if self.ignore_new_domain: # Modify dialog acts to ignore new domains
            dialog_acts = [[i,d,s,v] for i,d,s,v in dialog_acts if d in domain_memory]
        return PolicyPPNOutput(system_action=dialog_acts, processed_da=processed_da)

    def _through_module_output(self, module_output: PolicyOutput):
        return module_output