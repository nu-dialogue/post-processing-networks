import re
from ppn import AbstractPPN
from nlu.joint_bert.nlu import MyBERTNLU
from nlu import NLUOutput, NLUPPNOutput
from util import get_logger
logger = get_logger(__name__)

class MyBERTNLUPPN(AbstractPPN):
    def __init__(self, module: MyBERTNLU, system_state_dim: int) -> None:
        super().__init__(module=module, system_state_dim=system_state_dim)

    def _prepare_vocab(self):
        self.__intent_vocab = self.module.intent_vocab
        self.__tag_vocab = []
        for bio_intent in self.module.tag_vocab:
            if bio_intent.startswith("B-"):
                self.__tag_vocab.append("{}*value".format(bio_intent.replace("B-", "")))
        self.__id2da = {}
        self.__da2id = {}
        for i, da in enumerate(self.__intent_vocab + self.__tag_vocab):
            self.__id2da[i] = da
            self.__da2id[da] = i
        
    def _set_model_params(self):
        self.model_params.update({
            "obs_dim": len(self.__id2da) + self.system_state_dim,
            "act_dim": len(self.__id2da)
        })

    def _vectorize(self, module_output: NLUOutput):
        user_action = module_output.user_action
        dialog_acts_vec = [0 for _ in range(len(self.__id2da))]
        domain_memory = set() # Set to hold domain information for the base module's output
        value_memory = {} # Dictionary for restoring values of the tags
        for intent, domain, slot, value in user_action:
            domain_memory.add(domain)
            da_key = "{}-{}+{}".format(domain, intent, slot)
            intent_da = "{}*{}".format(da_key, value)
            tag_da = "{}*value".format(da_key)
            if intent_da in self.__da2id:
                dialog_acts_vec[self.__da2id[intent_da]] = 1
            elif tag_da in self.__da2id:
                dialog_acts_vec[self.__da2id[tag_da]] = 1
                value_memory[da_key] = value
            else:
                logger.info("{}:: {} is unknown dialog act.".format(self.type, tag_da))
        items = {"domain_memory": domain_memory,"value_memory": value_memory}
        return dialog_acts_vec, items

    def _devectorize(self, processed_da_vec, **items):
        domain_memory = items["domain_memory"]
        value_memory = items["value_memory"]

        diglog_acts, processed_da = [], []
        for i in range(len(processed_da_vec)):
            if not processed_da_vec[i]:
                continue
            processed_da.append(self.__id2da[i])
            domain_intent_slot, value = self.__id2da[i].split('*')
            domain, intent, slot = re.split('[+-]', domain_intent_slot)
            if self.ignore_new_domain and domain not in domain_memory:
                continue
            if value == "value":
                if domain_intent_slot in value_memory:
                    # When the slot is da with value and the NLU was also outputting it
                    value = value_memory[domain_intent_slot]
                else:
                    continue
            diglog_acts.append([intent, domain, slot, value])
        return NLUPPNOutput(user_action=diglog_acts, processed_da=processed_da)

    def _through_module_output(self, module_output: NLUOutput):
        return module_output