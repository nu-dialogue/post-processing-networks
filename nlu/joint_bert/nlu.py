import numpy as np
import re
from unidecode import unidecode
from convlab2.nlu.jointBERT.multiwoz import BERTNLU

from nlu import AbstractNLU, NLUOutput
from nlu.joint_bert.utils import recover_intent

class MyBERTNLU(AbstractNLU):
    def __init__(self, module_type, module_config) -> None:
        super().__init__(module_type, module_config)
        assert self.type == "nlu"
        assert self.name == "bert"
        self.bert_nlu = BERTNLU(mode="usr", config_file="multiwoz_usr_context.json",
                                model_file="https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_usr_context.zip")
        self.intent_vocab = self.bert_nlu.dataloader.intent_vocab
        self.tag_vocab = self.bert_nlu.dataloader.tag_vocab
        self.intent_scores_dim = len(self.bert_nlu.dataloader.id2intent)
        self.tag_scores_dim = len(self.bert_nlu.dataloader.id2tag)

    @property
    def module_state_dim(self):
        return self.intent_scores_dim + self.tag_scores_dim

    def module_state_vector(self):
        assert self.intent_scores_dim == len(self.intent_scores)
        assert self.tag_scores_dim == len(self.tag_scores)
        state_vector = np.array(self.intent_scores + self.tag_scores)
        return state_vector

    def init_session(self):
        self.bert_nlu.init_session()
        # Initial module state
        # List of probabilities (sigmoid(logits)) for each predicted intent
        self.intent_scores = [0 for _ in range(len(self.bert_nlu.dataloader.id2intent))]
         # List of probabilities (softmax(logits)) for each predicted slot
        self.tag_scores = [0 for _ in range(len(self.bert_nlu.dataloader.id2tag))]

    def predict(self, observation, context=[]):
        # Note: spacy cannot tokenize 'id' or 'Id' correctly.
        observation = re.sub(r'\b(id|Id)\b', 'ID', observation)
        # tokenization first, very important!
        ori_word_seq = [token.text for token in self.bert_nlu.nlp(unidecode(observation)) if token.text.strip()]
        ori_tag_seq = ['O'] * len(ori_word_seq)
        if self.bert_nlu.use_context:
            if len(context) > 0 and type(context[0]) is list and len(context[0]) > 1:
                context = [item[1] for item in context]
            context_seq = self.bert_nlu.dataloader.tokenizer.encode('[CLS] ' + ' [SEP] '.join(context[-3:]))
            context_seq = context_seq[:512]
        else:
            context_seq = self.bert_nlu.dataloader.tokenizer.encode('[CLS]')
        intents = []
        da = {}

        word_seq, tag_seq, new2ori = self.bert_nlu.dataloader.bert_tokenize(ori_word_seq, ori_tag_seq)
        word_seq = word_seq[:510]
        tag_seq = tag_seq[:510]
        batch_data = [[ori_word_seq, ori_tag_seq, intents, da, context_seq,
                       new2ori, word_seq, self.bert_nlu.dataloader.seq_tag2id(tag_seq), self.bert_nlu.dataloader.seq_intent2id(intents)]]

        pad_batch = self.bert_nlu.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to(self.bert_nlu.model.device) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        slot_logits, intent_logits = self.bert_nlu.model.forward(word_seq_tensor, word_mask_tensor,
                                                        context_seq_tensor=context_seq_tensor,
                                                        context_mask_tensor=context_mask_tensor)

        recovered = recover_intent(self.bert_nlu.dataloader, intent_logits[0], slot_logits[0], tag_mask_tensor[0], batch_data[0][0], batch_data[0][-4])
        dialog_acts = recovered["intents"] + recovered["tag_intents"]
        # dialog_act_probas = recovered["intent_probas"] + recovered["tag_intent_probas"]

        self.intent_scores = recovered["intent_entire_probas"]
        self.tag_scores = recovered["tag_intent_entire_probas"]

        return NLUOutput(user_action=dialog_acts)