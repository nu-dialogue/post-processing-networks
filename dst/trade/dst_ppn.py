from copy import deepcopy
import numpy as np
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.util.multiwoz.state import default_state

from dst import DSTOutput, DSTPPNOutput
from ppn import AbstractPPN
from dst.trade.dst import MyTRADEDST

class MyTRADEDSTPPN(AbstractPPN):
    def __init__(self, module: MyTRADEDST, system_state_dim: int):
        super().__init__(module, system_state_dim)

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
        request_state_vocab = []
        for key_Domain in self.module.trade_dst.det_dic.values():
            if key_Domain not in request_state_vocab:
                request_state_vocab.append(key_Domain)
        self.__request_state_dim = len(request_state_vocab)
        self.__id2rs_slot = {}
        self.__rs_slot2id = {}
        for i, slot in enumerate(request_state_vocab):
            self.__id2rs_slot[i] = slot
            self.__rs_slot2id[slot] = i

    def _set_model_params(self):
        self.model_params.update({
            "obs_dim": self.__belief_state_dim + self.system_state_dim,
            "act_dim": self.__belief_state_dim
        })

    def _vectorize(self, module_output: DSTOutput):
        belief_state = module_output.state["belief_state"]
        # request_state = module_output.state["request_state"]

        bs_vec = [0 for _ in range(self.__belief_state_dim)]
        for domain, domain_bs in belief_state.items():
            for slot, value in domain_bs["semi"].items():
                if value:
                    slot_key = "{}-semi-{}".format(domain.lower(), slot)
                    bs_vec[self.__bs_slot2id[slot_key]] = 1

        # rs_vec = [0 for _ in range(self.__request_state_dim)]
        # for domain, domain_rs in request_state.items():
        #     for slot in domain_rs:
        #         key_Domain = "{}-{}".format(slot, domain)
        #         rs_vec[self.__rs_slot2id[key_Domain]] = 1

        return bs_vec, {"dst_state": deepcopy(module_output.state)}

    def _devectorize(self, processed_module_output_vec, **items):
        assert processed_module_output_vec.shape[0] == self.__belief_state_dim # + self.__request_state_dim
        dst_state = items["dst_state"]

        belief_state = dst_state["belief_state"]
        deleted_bs = []
        for i, is_active in enumerate(processed_module_output_vec[:self.__belief_state_dim]):
            slot_key = self.__id2bs_slot[i]
            if is_active:
                pass
            else:
                domain, semi, slot = slot_key.split("-")
                try:
                    if belief_state[domain][semi][slot]:
                        deleted_bs.append(slot_key)
                    belief_state[domain][semi][slot] = ""
                except KeyError:
                    pass
        
        # request_state = dst_state["request_state"]
        # added_rs, deleted_rs = [], []
        # for i, is_active in enumerate(processed_module_output_vec[-self.__request_state_dim:]):
        #     key_Domain = self.__id2rs_slot[i]
        #     slot, Domain = key_Domain.split("-")
        #     if is_active:
        #         if Domain not in request_state:
        #             request_state[Domain] = {}
        #         if slot not in request_state[Domain]:
        #             added_rs.append(key_Domain)
        #         request_state[Domain][slot] = 0
        #     else:
        #         try:
        #             if request_state[Domain][slot] == 0:
        #                 deleted_rs.append(key_Domain)
        #             del request_state[Domain][slot]
        #         except KeyError:
        #             pass


        dst_state["belief_state"] = deepcopy(belief_state)
        # dst_state["request_state"] = deepcopy(request_state)
        return DSTPPNOutput(state=dst_state, deleted_bs=deleted_bs, added_rs=[], deleted_rs=[])

    def _through_module_output(self, module_output: DSTOutput):
        return module_output