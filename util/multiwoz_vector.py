import os
import numpy as np
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.util.multiwoz.lexicalize import delexicalize_da, flat_da, deflat_da, lexicalize_da

from util.path import ROOT_DPATH
MULTIWOZ_DATA_DPATH = os.path.join(ROOT_DPATH, "ConvLab-2/data/multiwoz")

def process_str_action(action):
    if action and isinstance(action[0], str):
        action = []
    return action

class FixedMultiWozVector(MultiWozVector):
    def __init__(self):
        super().__init__(voc_file=os.path.join(MULTIWOZ_DATA_DPATH, 'sys_da_voc.txt'),
                         voc_opp_file=os.path.join(MULTIWOZ_DATA_DPATH, 'usr_da_voc.txt'),
                         intent_file=os.path.join(MULTIWOZ_DATA_DPATH, 'trackable_intent.json'))

    def state_vectorize(self, state):
        self.state = state['belief_state']

        # when character is sys, to help query database when da is booking-book
        # update current domain according to user action
        if self.character == 'sys':
            action = state['user_action']
            action = process_str_action(action)
            for act in action:
                intent, domain, slot, value = act
                if domain in self.db_domains:
                    self.cur_domain = domain

        action = state['user_action'] if self.character == 'sys' else state['system_action']
        action = process_str_action(action)
        opp_action = delexicalize_da(action, self.requestable)
        opp_action = flat_da(opp_action)
        opp_act_vec = np.zeros(self.da_opp_dim)
        for da in opp_action:
            if da in self.opp2vec:
                opp_act_vec[self.opp2vec[da]] = 1.

        action = state['system_action'] if self.character == 'sys' else state['user_action']
        action = process_str_action(action)
        action = delexicalize_da(action, self.requestable)
        action = flat_da(action)
        last_act_vec = np.zeros(self.da_dim)
        for da in action:
            if da in self.act2vec:
                last_act_vec[self.act2vec[da]] = 1.

        belief_state = np.zeros(self.belief_state_dim)
        i = 0
        for domain in self.belief_domains:
            for slot, value in state['belief_state'][domain.lower()]['semi'].items():
                if value:
                    belief_state[i] = 1.
                i += 1

        book = np.zeros(len(self.db_domains))
        for i, domain in enumerate(self.db_domains):
            if state['belief_state'][domain.lower()]['book']['booked']:
                book[i] = 1.

        degree = self.pointer(state['belief_state'])

        final = 1. if state['terminated'] else 0.

        state_vec = np.r_[opp_act_vec, last_act_vec, belief_state, book, degree, final]
        assert len(state_vec) == self.state_dim
        return state_vec