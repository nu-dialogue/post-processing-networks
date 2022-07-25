import re
from ppn import AbstractPPN
from nlu.svm.nlu import MySVMNLU
from nlu import NLUOutput, NLUPPNOutput
from util import get_logger
logger = get_logger(__name__)

class MySVMNLUPPN(AbstractPPN):
    def __init__(self, module: MySVMNLU, system_state_dim: int):
        super().__init__(module, system_state_dim)

    def _prepare_vocab(self):
        count = 0
        self.__id2da = {}

        self.__da_wo_value2id = {}
        for request, slot in self.module.tuples_wo_value:
            domain, slot = slot.split('-')
            da_key = "{}-{}+{}*?".format(domain, request.capitalize(), slot)
            if da_key in self.__da_wo_value2id:
                continue
            self.__da_wo_value2id[da_key] = count
            self.__id2da[count] = da_key
            count += 1
        assert len(self.__id2da) == len(self.__da_wo_value2id)
        
        self.__da_wo_generic2id = {}
        for intent, slot, value in self.module.tuples_wo_generic:
            domain, intent = intent.split('-')
            da_key = "{}-{}+{}*{}".format(domain, intent, slot, value)
            if da_key in self.__da_wo_generic2id:
                continue
            self.__da_wo_generic2id[da_key] = count
            self.__id2da[count] = da_key
            count += 1
        assert len(self.__id2da) + len(self.__da_wo_value2id) + len(self.__da_wo_generic2id)

        self.__da_w_generic2id = {}
        for intent, slot in self.module.tuples_w_generic:
            domain, intent = intent.split('-')
            da_key = "{}-{}+{}*generic".format(domain, intent, slot)
            if da_key in self.__da_w_generic2id:
                continue
            self.__da_w_generic2id[da_key] = count
            self.__id2da[count] = da_key
            count += 1
        assert len(self.__id2da) == len(self.__da_wo_value2id) + len(self.__da_wo_generic2id) + len(self.__da_w_generic2id)
        

    def _set_model_params(self):
        self.model_params.update({
            "obs_dim": len(self.__id2da) + self.system_state_dim,
            "act_dim": len(self.__id2da)
        })

    def _vectorize(self, module_output: NLUOutput):
        user_action = module_output.user_action
        da_vec = [0 for _ in range(len(self.__id2da))]
        domain_memory = set()
        value_memory = {}
        for intent, domain, slot, value in user_action:
            domain_memory.add(domain)
            domain_intent_slot = "{}-{}+{}".format(domain, intent, slot)
            da_key_wo_value = domain_intent_slot + "*?"
            da_key_wo_generic = domain_intent_slot + "*" + value
            da_key_w_generic = domain_intent_slot + "*generic"
            if da_key_wo_value in self.__da_wo_value2id:
                da_vec[self.__da_wo_value2id[da_key_wo_value]] = 1
            elif da_key_wo_generic in self.__da_wo_generic2id:
                da_vec[self.__da_wo_generic2id[da_key_wo_generic]] = 1
            elif da_key_w_generic in self.__da_w_generic2id:
                da_vec[self.__da_w_generic2id[da_key_w_generic]] = 1
                value_memory[domain_intent_slot] = value
            else:
                logger.info("{}:: {} is unknown dialog act.".format(self.type, da_key_wo_generic))
        items = {"domain_memory": domain_memory,"value_memory": value_memory}
        return da_vec, items

    def _devectorize(self, ppn_output_vec, **items):
        domain_memory = items["domain_memory"]
        value_memory = items["value_memory"]

        diglog_acts, processed_da = [], []
        for i in range(len(ppn_output_vec)):
            if not ppn_output_vec[i]:
                continue
            da_key = self.__id2da[i]
            processed_da.append(da_key)
            domain_intent_slot, value = da_key.split('*')
            domain, intent, slot = re.split('[+-]', domain_intent_slot)
            if self.ignore_new_domain and domain not in domain_memory:
                continue
            if value == "generic":
                if domain_intent_slot in value_memory:
                    # When the slot is da with value and the NLU was also outputting it
                    value = value_memory[domain_intent_slot]
                    # logger.info("restore value {}".format(value))
                else:
                    continue
            diglog_acts.append([intent, domain, slot, value])
        return NLUPPNOutput(user_action=diglog_acts, processed_da=processed_da)
    
    def _through_module_output(self, module_output: NLUOutput):
        return module_output