import numpy as np
import json
from collections import OrderedDict
from convlab2.nlg.template.multiwoz import TemplateNLG

from nlg import AbstractNLG, NLGOutput
from util import get_logger
logger = get_logger(__name__)

class MyTemplateNLG(AbstractNLG):
    def __init__(self, module_type, module_config) -> None:
        super().__init__(module_type, module_config)
        assert self.type == "nlg"
        assert self.name == "template"
        self.template_nlg = TemplateNLG(is_user=False)
        self.mode_list = ["manual", "auto", "auto_manual"]
        self.default_mode = "manual"
    
    @property
    def module_state_dim(self) -> int:
        return len(self.mode_list)
    
    def module_state_vector(self) -> np.ndarray:
        mode_onehot = np.zeros(self.module_state_dim)
        mode_onehot[self.mode_list.index(self.default_mode)] = 1
        return mode_onehot

    def init_session(self) -> None:
        self.template_nlg.init_session()

    def generate(self, dialog_acts: list) -> dict:
        dialog_acts = self.template_nlg.sorted_dialog_act(dialog_acts)
        action = OrderedDict()
        for intent, domain, slot, value in dialog_acts:
            k = '-'.join([domain.lower(), intent.lower()])
            action.setdefault(k, [])
            action[k].append([slot.lower(), value])
        dialog_acts = action
        
        responses = {"manual": "", "auto": "", "auto_manual": ""}
        try:
            # manual template
            responses["manual"] = self.template_nlg._manual_generate(dialog_acts, self.template_nlg.manual_system_template)

            # auto template
            responses["auto"] = self.template_nlg._auto_generate(dialog_acts, self.template_nlg.auto_system_template)

            # auto manual template
            responses["auto_manual"] = self.template_nlg._auto_generate(dialog_acts, self.template_nlg.auto_system_template)
            if responses["auto_manual"] == 'None':
                responses["auto_manual"] = self.template_nlg._manual_generate(dialog_acts, self.template_nlg.manual_system_template)

        except Exception as e:
            logger.error('Error in processing:')
            logger.error(json.dumps(dialog_acts,indent=4))
            raise e

        return NLGOutput(system_response=responses[self.default_mode], system_responses=responses)
    