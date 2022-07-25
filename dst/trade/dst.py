import os
import numpy as np
import torch
from copy import deepcopy
from convlab2.dst.trade.multiwoz import TRADE
from convlab2.dst.trade.multiwoz.utils.config import args

from dst import AbstractDST, DSTOutput
from util import FixedMultiWozVector

class MyTRADEDST(AbstractDST):
    def __init__(self, module_type, module_config) -> None:
        super().__init__(module_type, module_config)
        assert self.type == "dst"
        assert self.name == "trade"

        self.trade_dst = TRADE()
        self.vector = FixedMultiWozVector()
        
    @property
    def module_state_dim(self):
        return self.vector.state_dim
    
    def module_state_vector(self):
        state_vector = self.vector.state_vectorize(self.trade_dst.state)
        return state_vector

    def init_session(self):
        self.trade_dst.init_session()
        self.current_gate = np.full(len(self.trade_dst.slots), 2)

    def replace_state(self, key, value):
        if key == "state":
            self.trade_dst.state = value
        elif key in self.trade_dst.state:
            self.trade_dst.state[key] = value
        else:
            raise KeyError("{} is not in rule dst keys ({})".format(key, self.trade_dst.state.keys()))
        
    def append_history(self, data):
        self.trade_dst.state["history"].append(data)

    def update(self, action):
        state = self.trade_dst.update(user_act=action)  
        return DSTOutput(state=state)