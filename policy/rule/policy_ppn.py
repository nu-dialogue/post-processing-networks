import numpy as np
from policy import PolicyOutput, PolicyPPNOutput
from ppn import AbstractPPN
from policy.rule.policy import MyRulePolicy

class MyRulePolicyPPN(AbstractPPN):
    def __init__(self, module: MyRulePolicy, system_state_dim: int):
        super().__init__(module=module, system_state_dim=system_state_dim)

    def _prepare_vocab(self):
        self.__vector = self.module.vector

    def _set_model_params(self):
        self.model_params.update({
            "obs_dim": len(self.__vector.da_voc) + self.system_state_dim,
            "act_dim": len(self.__vector.da_voc)
        })

    def _vectorize(self, module_output):
        system_action = module_output.system_action
        value_memory, domain_memory = dict(), set()
        for i,d,s,v in system_action:
            value_memory[f'{i}-{d}-{s}'] = v
            domain_memory.add(d)

        da_vec = self.__vector.action_vectorize(system_action)
        return da_vec, {"domain_memory": domain_memory, "value_memory": value_memory}

    def _devectorize(self, ppn_output_vec, **items):
        value_memory = items["value_memory"]
        domain_memory = items["domain_memory"]

        system_action = []
        processed_da = []
        dialog_acts = self.__vector.action_devectorize(ppn_output_vec)
        for i,d,s,v in dialog_acts:
            processed_da.append(f"{d}-{i}-{s}-{v}")
            if self.ignore_new_domain and d in domain_memory:
                continue
            da_key = f"{i}-{d}-{s}"
            if da_key in value_memory:
                v = value_memory[da_key]
            system_action.append([i,d,s,v])
        return PolicyPPNOutput(system_action=system_action, processed_da=processed_da)

    def _through_module_output(self, module_output: PolicyOutput):
        return module_output