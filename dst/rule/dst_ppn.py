from copy import deepcopy
import numpy as np
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_USR_DA

from dst import DSTOutput, DSTPPNOutput
from ppn import AbstractPPN
from dst.rule.dst import MyRuleDST

class MyRuleDSTPPN(AbstractPPN):
    def __init__(self, module: MyRuleDST, system_state_dim: int) -> None:
        super().__init__(module=module, system_state_dim=system_state_dim)

    def _prepare_vocab(self):
        # belief state
        self.__belief_domains: MultiWozVector = self.module.vector.belief_domains
        self.__belief_state_dim = self.module.vector.belief_state_dim
        self.__default_bs = default_state()["belief_state"]
        belief_state_vocab = []
        for domain in self.__belief_domains:
            for slot in self.__default_bs[domain.lower()]["semi"]:
                belief_state_vocab.append("{}-semi-{}".format(domain.lower(), slot))
        assert len(belief_state_vocab) == self.__belief_state_dim
        self.__id2bs_slot = {}
        self.__bs_slot2id = {}
        for i, slot in enumerate(belief_state_vocab):
            self.__id2bs_slot[i] = slot
            self.__bs_slot2id[slot] = i

        # request state
        # request_state_vocab = []
        # for Domain, dic in REF_USR_DA:
        #     for key, value in dic.items():
        #         assert '-' not in key
        #         key_Domain = "{}-{}".format(key.lower(), Domain)
        #         value_Domain = "{}-{}".format(value.lower(), Domain)
        #         if key_Domain not in request_state_vocab:
        #             request_state_vocab.append(key_Domain)
        #         if value_Domain not in request_state_vocab:
        #             request_state_vocab.append(value_Domain)
        # self.__request_state_dim = len(request_state_vocab)
        # self.__id2rs_slot = {}
        # self.__rs_slot2id = {}
        # for i, slot in enumerate(request_state_vocab):
        #     self.__id2rs_slot[i] = slot
        #     self.__rs_slot2id[slot] = i
        self.__request_state_dim = 0

    def _set_model_params(self):
        self.model_params.update({
            "obs_dim": self.__belief_state_dim + self.__request_state_dim + self.system_state_dim,
            "act_dim": self.__belief_state_dim + self.__request_state_dim
        })

    def _vectorize(self, module_output: DSTOutput):
        belief_state = module_output.state["belief_state"]

        bs_vec = [0 for _ in range(len(self.__id2bs_slot))]
        for domain, domain_bs in belief_state.items():
            for slot, value in domain_bs["semi"].items():
                if value:
                    slot_key = "{}-semi-{}".format(domain.lower(), slot)
                    bs_vec[self.__bs_slot2id[slot_key]] = 1
        return bs_vec, {"dst_state": deepcopy(module_output.state)}

    def _devectorize(self, processed_module_output_vec, **items):
        dst_state = items["dst_state"]

        belief_state = deepcopy(dst_state["belief_state"])
        processed_bs, deleted_bs = [], []
        for i, is_active in enumerate(processed_module_output_vec):
            slot_key = self.__id2bs_slot[i]
            if is_active:
                processed_bs.append(slot_key)
            else:
                domain, semi, slot = slot_key.split("-")
                try:
                    if belief_state[domain][semi][slot]:
                        deleted_bs.append(slot_key)
                    belief_state[domain][semi][slot] = ""
                except:
                    pass

        dst_state["belief_state"] = deepcopy(belief_state)
        return DSTPPNOutput(state=dst_state, deleted_bs=deleted_bs, added_rs=[], deleted_rs=[])

    def _through_module_output(self, module_output: DSTOutput):
        return module_output