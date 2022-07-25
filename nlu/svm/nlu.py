import numpy as np
from convlab2.nlu.svm.multiwoz import SVMNLU

from nlu import AbstractNLU, NLUOutput

class MySVMNLU(AbstractNLU):
    def __init__(self, module_type, module_config) -> None:
        super().__init__(module_type, module_config)
        assert self.type == "nlu"
        assert self.name == "svm"
        self.svm_nlu = SVMNLU(mode='usr')
        self.tuples_wo_value = []
        self.tuples_wo_generic = []
        self.tuples_w_generic = []
        for tuple_ in self.svm_nlu.c.tuples.all_tuples:
            if len(tuple_) == 2:
                assert tuple_[0] == "request"
                self.tuples_wo_value.append(tuple_)
            elif isinstance(tuple_[-1], str):
                self.tuples_wo_generic.append((tuple_[0], tuple_[1], tuple_[2].lower()))
            else:
                self.tuples_w_generic.append(tuple_[:-1])

    @property
    def module_state_dim(self):
        return len(self.tuples_wo_value) + len(self.tuples_wo_generic) + len(self.tuples_w_generic)
    
    def module_state_vector(self):
        vec_wo_value = np.zeros(len(self.tuples_wo_value))
        vec_wo_generic = np.zeros(len(self.tuples_wo_generic))
        vec_w_generic = np.zeros(len(self.tuples_w_generic))
        for slu_hyp in self.slu_hyps:
            score = slu_hyp['score']
            for da in slu_hyp['slu-hyp']:
                act, slots = da['act'], da['slots'][0]
                if slots[0] == 'slot':
                    assert act == "request"
                    i = self.tuples_wo_value.index((act, slots[1]))
                    if not vec_wo_value[i]:
                        vec_wo_value[i] = score
                else:
                    if (act, slots[0], slots[1]) in self.tuples_wo_generic:
                        i = self.tuples_wo_generic.index((act, slots[0], slots[1]))
                        if not vec_wo_generic[i]:
                            vec_wo_generic[i] = score
                    else:
                        i = self.tuples_w_generic.index((act, slots[0]))
                        if not vec_w_generic[i]:
                            vec_w_generic[i] = score
        return np.r_[vec_wo_value, vec_wo_generic, vec_w_generic]
    
    def init_session(self) -> None:
        self.svm_nlu.init_session()
        self.slu_hyps = []

    def predict(self, observation: str, context: list) -> NLUOutput:
        sentinfo = {
            "turn-id": 0,
            "asr-hyps": [
                    {
                        "asr-hyp": observation,
                        "score": 0
                    }
                ]
            }
        self.slu_hyps = self.svm_nlu.c.decode_sent(sentinfo, self.svm_nlu.config.get("decode", "output"))
        act_list = []
        for hyp in self.slu_hyps:
            if hyp['slu-hyp']:
                act_list = hyp['slu-hyp']
                break
        dialog_act = {}
        for act in act_list:
            intent = act['act']
            if intent=='request':
                domain, slot = act['slots'][0][1].split('-')
                intent = domain+'-'+intent.capitalize()
                dialog_act.setdefault(intent,[])
                dialog_act[intent].append([slot,'?'])
            else:
                dialog_act.setdefault(intent, [])
                dialog_act[intent].append(act['slots'][0])
        tuples = []
        for domain_intent, svs in dialog_act.items():
            for slot, value in svs:
                domain, intent = domain_intent.split('-')
                tuples.append([intent, domain, slot, value])
        return NLUOutput(user_action=tuples)