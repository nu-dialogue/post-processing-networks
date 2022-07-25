import re
import torch

def is_slot_da(da):
    tag_da = {'Inform', 'Select', 'Recommend', 'NoOffer', 'NoBook', 'OfferBook', 'OfferBooked', 'Book'}
    not_tag_slot = {'Internet', 'Parking', 'none'}
    if da[0].split('-')[1] in tag_da and da[1] not in not_tag_slot:
        return True
    return False

def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        predicts = [[x[0], x[1], x[2].lower()] for x in predicts]
        labels = item['golden']
        labels = [[x[0], x[1], x[2].lower()] for x in labels]
        for ele in predicts:
            if ele in labels:
                TP += 1
            else:
                FP += 1
        for ele in labels:
            if ele not in predicts:
                FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, F1

def tag2triples(dataloader, word_seq, tag_seq):
    try:
        assert len(word_seq)==len(tag_seq)
    except AssertionError:
        print(word_seq)
        print(tag_seq)
        raise
    triples = []
    proba_triples = []
    entire_probas = [0 for _ in range(len(dataloader.id2tag))]
    i = 0
    while i < len(tag_seq):
        tag, tag_proba = tag_seq[i]
        if tag.startswith('B'):
            entire_probas[dataloader.tag2id[tag]] = tag_proba
            domain_intent, slot = tag[2:].split('+')
            domain, intent = domain_intent.split("-")
            value = word_seq[i]
            proba_list = [tag_proba]
            j = i + 1
            while j < len(tag_seq):
                next_tag, next_tag_proba = tag_seq[j]
                if next_tag.startswith('I') and next_tag[2:] == tag[2:]:
                    entire_probas[dataloader.tag2id[next_tag]] = next_tag_proba
                    value += ' ' + word_seq[j]
                    proba_list.append(next_tag_proba)
                    i += 1
                    j += 1
                else:
                    break
            triples.append([intent, domain, slot, value])
            proba_triples.append([intent, domain, slot, sum(proba_list)/len(proba_list)])
        i += 1
    return triples, proba_triples, entire_probas

def recover_intent(dataloader, intent_logits, tag_logits, tag_mask_tensor, ori_word_seq, new2ori):
    # >>> intent w/o slot >>>
    intents = []
    intent_probas = []
    intent_entire_probas = []
    assert dataloader.intent_dim == intent_logits.shape[0]
    for j in range(dataloader.intent_dim):
        logit = intent_logits[j]
        proba = torch.sigmoid(logit)
        if logit.item() > 0:
            domain_intent, slot, value = re.split('[+*]', dataloader.id2intent[j])
            domain, intent = domain_intent.split('-')
            intents.append([intent, domain, slot, value])
            intent_probas.append([intent, domain, slot, proba.item()])
        intent_entire_probas.append(proba.item())
    # <<< intent w/o slot <<<

    # >>> intent w slot >>>
    max_seq_len = tag_logits.size(0)
    tag_probas = torch.softmax(tag_logits, dim=1)
    tags = []
    for j in range(1, max_seq_len-1):
        if tag_mask_tensor[j] == 1:
            tag_proba, tag_id = torch.max( tag_probas[j], dim=-1)
            tags.append([dataloader.id2tag[tag_id.item()], tag_proba.item()])
    recover_tags = []
    for i, tag in enumerate(tags):
        if new2ori[i] >= len(recover_tags):
            recover_tags.append(tag)
    tag_intents, tag_intent_probas, tag_intent_entire_probas = tag2triples(dataloader, ori_word_seq, recover_tags)
    # <<< intent w slot <<<

    ret = {
        "intents": intents,
        "tag_intents": tag_intents,
        "intent_probas": intent_probas,
        "tag_intent_probas": tag_intent_probas,
        "intent_entire_probas": intent_entire_probas,
        "tag_intent_entire_probas": tag_intent_entire_probas
    }
    return ret