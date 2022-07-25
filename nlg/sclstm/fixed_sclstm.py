from copy import deepcopy
import torch
from convlab2.nlg.sclstm.multiwoz import SCLSTM

def lexicalize(meta, delex):
    # replace the placeholder with entities
    recover = []
    for sen in delex:
        counter = {}
        words = sen.split()
        for word in words:
            if word.startswith('slot-'):
                flag = True
                _, domain, intent, slot_type = word.split('-')
                da = domain.capitalize() + '-' + intent.capitalize()
                if da in meta:
                    key = da + '-' + slot_type.capitalize()
                    for pair in meta[da]:
                        if (pair[0].lower() == slot_type) and (
                                (key not in counter) or (counter[key] == int(pair[1]) - 1)):
                            sen = sen.replace(word, pair[2], 1)
                            counter[key] = int(pair[1])
                            flag = False
                            break
                if flag:
                    sen = sen.replace(word, '', 1)
        recover.append(sen)
        break
    return recover
class FixedSCLSTM(SCLSTM):
    def __init__(self):
        super().__init__(use_cuda=True)

    def generate_with_beamsizes(self, meta, beamsize_list):
        action = {}
        for intent, domain, slot, value in meta:
            k = '-'.join([domain, intent])
            action.setdefault(k, [])
            action[k].append([slot, value])
        meta = action

        for k, v in meta.items():
            domain, intent = k.split('-')
            if intent == "Request":
                for pair in v:
                    if type(pair[1]) != str:
                        pair[1] = str(pair[1])
                    pair.insert(1, '?')
            else:
                counter = {}
                for pair in v:
                    if type(pair[1]) != str:
                        pair[1] = str(pair[1])
                    if pair[0] == 'none':
                        pair.insert(1, 'none')
                    else:
                        if pair[0] in counter:
                            counter[pair[0]] += 1
                        else:
                            counter[pair[0]] = 1
                        pair.insert(1, str(counter[pair[0]]))

        # remove invalid dialog act
        meta_ = deepcopy(meta)
        for k, v in meta.items():
            for triple in v:
                voc = 'd-a-s-v:' + k + '-' + triple[0] + '-' + triple[1]
                if voc not in self.dataset.cardinality:
                    meta_[k].remove(triple)
            if not meta_[k]:
                del (meta_[k])
        meta = meta_

        # mapping the inputs
        do_idx, da_idx, sv_idx, featStr = self.dataset.getFeatIdx(meta)
        do_cond = [1 if i in do_idx else 0 for i in range(self.dataset.do_size)]  # domain condition
        da_cond = [1 if i in da_idx else 0 for i in range(self.dataset.da_size)]  # dial act condition
        sv_cond = [1 if i in sv_idx else 0 for i in range(self.dataset.sv_size)]  # slot/value condition
        feats = [do_cond + da_cond + sv_cond]

        feats_var = torch.FloatTensor(feats)
        if self.USE_CUDA:
            feats_var = feats_var.cuda()

        response_dict = {}
        for beamsize in beamsize_list:
            delex = self.model.generate(self.dataset, feats_var, beamsize)[0]
            response_dict[beamsize] = lexicalize(meta, delex)[0]
    
        return response_dict