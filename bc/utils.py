import torch

def comupte_F1(act_weights, target_acts, score_history):
    def f1(a, target):
        TP, FP, FN = 0, 0, 0
        real = target.nonzero().tolist()
        predict = a.nonzero().tolist()
        # print("real", real)
        # print("predict", predict)
        # input()
        for item in real:
            if item in predict:
                TP += 1
            else:
                FN += 1
        for item in predict:
            if item not in real:
                FP += 1
        return TP, FP, FN

    if score_history is None and (act_weights is None or target_acts is None):
        raise Exception("Input 'score history' or 'act_weights-target_acts pair' for evaluation.")
    if score_history is None:
        pred_acts = act_weights.ge(0)
        return f1(pred_acts, target_acts)
    else:
        a_TP, a_FP, a_FN = [0]*3
        for TP, FP, FN in score_history:
            a_TP += TP
            a_FP += FP
            a_FN += FN
        prec, rec, F1 = 0, 0, 0
        if a_TP or a_FP:
            prec = a_TP / (a_TP + a_FP)
        if a_TP or a_FN:
            rec = a_TP / (a_TP + a_FN)
        if prec or rec:
            F1 = 2 * prec * rec / (prec + rec)
    return {"F1": F1}

def comupte_accuracy(act_weights, target_acts, score_history):
    if score_history is None:
        pred_acts = act_weights.argmax(-1)
        return (pred_acts == target_acts).to(torch.float).mean().item()
    else:
        accuracy = sum(score_history)/len(score_history)
    return {"accuracy": accuracy}
