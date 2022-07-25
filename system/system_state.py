import json
import numpy as np
from pprint import pprint
import random
from util import get_logger
logger = get_logger(__name__)

class SystemState:
    """
    システム状態（全モジュールの状態を合わせたもの）のベクトルを管理するクラス
    """
    def __init__(self, nlu, dst, policy, nlg):
        self.nlu_state_dim = nlu.module_state_dim if nlu is not None else 0
        self.dst_state_dim = dst.module_state_dim if dst is not None else 0
        self.policy_state_dim = policy.module_state_dim if policy is not None else 0
        self.nlg_state_dim = nlg.module_state_dim if nlg is not None else 0
        
        self.dim = self.nlu_state_dim + self.dst_state_dim + self.policy_state_dim + self.nlg_state_dim

        # logger.info(json.dumps(self.__dict__, indent=4))

    def init_session(self):

        # History of each module state for current turn
        self.nlu_history = []
        self.dst_history = []
        self.policy_history = []
        self.nlg_history = []
        self.system_history = []

    def update(self, module_name, module_state):
        """Record a module's state"""
        # logger.info(module_name, module_state.shape)
        if module_name == "nlu":
            # logger.info(module_state)
            assert self.nlu_state_dim == len(module_state)
            self.nlu_history.append(module_state)

        elif module_name == "dst":
            assert self.dst_state_dim == len(module_state)
            self.dst_history.append(module_state)

        elif module_name == "policy":
            assert self.policy_state_dim == len(module_state)
            self.policy_history.append(module_state)

        elif module_name == "nlg":
            assert self.nlg_state_dim == len(module_state)
            self.nlg_history.append(module_state)

        else:
            raise Exception("{} is unknown module.".format(module_name))

    def latest(self, module_name="system"):
        """Return latest state of a module or all modules"""
        if module_name == "nlu":
            if self.nlu_state_dim:
                return self.nlu_history[-1]
            else:
                raise Exception("nlu module is not used.")
        
        elif module_name == "dst":
            if self.dst_state_dim:
                return self.dst_history[-1]
            else:
                raise Exception("dst module is not used.")
        
        elif module_name == "policy":
            if self.policy_state_dim:
                return self.policy_history[-1]
            else:
                raise Exception("policy module is not used.")
        
        elif module_name == "nlg":
            if self.nlg_state_dim:
                return self.nlg_history[-1]
            else:
                raise Exception("nlg module is not used.")
        
        elif module_name == "system":
            state = []
            if self.nlu_state_dim:
                state = np.r_[state, self.nlu_history[-1]]
            if self.dst_state_dim:
                state = np.r_[state, self.dst_history[-1]]
            if self.policy_state_dim:
                state = np.r_[state, self.policy_history[-1]]
            if self.nlg_state_dim:
                state = np.r_[state, self.nlg_history[-1]]
            return state
        
        else:
            raise Exception("{} is unknown module.".format(module_name))

    def random(self, module_name="system", active_ratio=0.3):
        """
        Generate a random binary vector as a pseudo system state
        """
        system_state = None
        if module_name == "nlu":
            system_state = [ random.random() <= active_ratio for _ in range(self.nlu_state_dim)]
        elif module_name == "dst":
            system_state = [ random.random() <= active_ratio for _ in range(self.dst_state_dim)]
        elif module_name == "policy":
            system_state = [ random.random() <= active_ratio for _ in range(self.policy_state_dim)]
        elif module_name == "system":
            system_state = [ random.random() <= active_ratio for _ in range(self.dim)]
        else:
            raise Exception("{} is unknown module.".format(module_name))
        return np.array(system_state)