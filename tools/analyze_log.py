from glob import glob
from copy import deepcopy
from collections import Counter
from tqdm import tqdm
import pandas as pd
import argparse
import json
import os
import sys
ROOT_DPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DPATH)

from util.plot import plot_mat, plot_sns, plot_sns_f1s

DOMAINS = ["attraction", "hotel", "restaurant", "train", "taxi", "hospital", "police"]

def load_main_score(log_dpath):
    total = []
    for log_fpath in tqdm(glob(os.path.join(log_dpath, "*.json"))):
        log = json.load(open(log_fpath))
        data = {
            "iteration_id": log["iteration_id"],
            "process_id": log["process_id"],
            "episode_id": log["episode_id"],
            "task_complete": log["task_complete"],
            "task_success": log["task_success"],
            "book_rate": log["book_rate"],
            "inform_precision": log["inform_precision"],
            "inform_recall": log["inform_recall"],
            "inform_F1": log["inform_F1"],
            "common_reward_avg": sum(log["common_reward_history"])/len(log["common_reward_history"]),
            "turn": log["turn"],
        }
        for domain in DOMAINS:
            data[f"domain_{domain}"] = domain in log["initial_goal"]
        total.append(data)

    df = pd.DataFrame(total).sort_values(by=["iteration_id", "process_id", "episode_id"])
    df["dialog_id"] = pd.RangeIndex(start=0, stop=len(df.index), step=1)
    df_by_domain = {domain: df[df[f"domain_{domain}"]] for domain in DOMAINS}
    return df, df_by_domain

def _compute_nlu_score(log, added_actions, deleted_actions):
    dialog_pred = {"recall": [], "precision": [], "f1": []}
    dialog_ppn_pred = {"recall": [], "precision": [], "f1": []}
    for i in range(log["turn"]):
        pred = {"TP":0, "FP":0, "FN": 0, "recall": 0, "precision": 0, "f1": 0}
        ppn_pred = {"TP":0, "FP":0, "FN": 0, "recall": 0, "precision": 0, "f1": 0}

        truth_actions = log["user_dialog"][i]["policy"]["user_atcion"]
        pred_actions = log["system_dialog"][i]["nlu"]["user_action"]
        ppn_pred_actions = log["system_dialog"][i]["nlu_ppn"]["user_action"]
        for truth_action in truth_actions:
            if truth_action in pred_actions:
                pred["TP"] += 1
            else:
                pred["FN"] += 1
            if truth_action in ppn_pred_actions:
                ppn_pred["TP"] += 1
            else:
                ppn_pred["FN"] += 1
        for pred_action in pred_actions:
            if pred_action not in truth_actions:
                pred["FP"] += 1
            if pred_action not in ppn_pred_actions:
                i_d_s = f"{pred_action[0]}-{pred_action[1]}-{pred_action[2]}"
                deleted_actions.append(i_d_s)
        for ppn_pred_action in ppn_pred_actions:
            if ppn_pred_action not in truth_actions:
                ppn_pred["FP"] += 1
            if ppn_pred_action not in pred_actions:
                i_d_s = f"{ppn_pred_action[0]}-{ppn_pred_action[1]}-{ppn_pred_action[2]}"
                added_actions.append(i_d_s)

        dialog_pred["recall"].append(0)
        dialog_pred["precision"].append(0)
        dialog_pred["f1"].append(0)
        if pred["TP"]:
            dialog_pred["recall"][-1] = pred["TP"] / (pred["TP"] + pred["FN"])
            dialog_pred["precision"][-1] = pred["TP"] / (pred["TP"] + pred["FP"])
            dialog_pred["f1"][-1] = 2*dialog_pred["recall"][-1]*dialog_pred["precision"][-1] / (dialog_pred["recall"][-1]+dialog_pred["precision"][-1])

        dialog_ppn_pred["recall"].append(0)
        dialog_ppn_pred["precision"].append(0)
        dialog_ppn_pred["f1"].append(0)
        if ppn_pred["TP"]:
            dialog_ppn_pred["recall"][-1] = ppn_pred["TP"] / (ppn_pred["TP"] + ppn_pred["FN"])
            dialog_ppn_pred["precision"][-1] = ppn_pred["TP"] / (ppn_pred["TP"] + ppn_pred["FP"])
            dialog_ppn_pred["f1"][-1] = 2*dialog_ppn_pred["recall"][-1]*dialog_ppn_pred["precision"][-1] / (dialog_ppn_pred["recall"][-1]+dialog_ppn_pred["precision"][-1])

    dialog_pred["recall"] = sum(dialog_pred["recall"]) / len(dialog_pred["recall"])
    dialog_pred["precision"] = sum(dialog_pred["precision"]) / len(dialog_pred["precision"])
    dialog_pred["f1"] = sum(dialog_pred["f1"]) / len(dialog_pred["f1"])

    dialog_ppn_pred["recall"] = sum(dialog_ppn_pred["recall"]) / len(dialog_ppn_pred["recall"])
    dialog_ppn_pred["precision"] = sum(dialog_ppn_pred["precision"]) / len(dialog_ppn_pred["precision"])
    dialog_ppn_pred["f1"] = sum(dialog_ppn_pred["f1"]) / len(dialog_ppn_pred["f1"])
    return dialog_pred, dialog_ppn_pred

def load_nlu_score(log_dpath, use_iteration_ids=[]):
    total = []
    added_actions = []
    deleted_actions = []
    for log_fpath in tqdm(glob(os.path.join(log_dpath, "*.json"))):
        iteration_id = int(os.path.basename(log_fpath).split("-")[0])
        if use_iteration_ids and iteration_id not in use_iteration_ids:
            continue
        log = json.load(open(log_fpath))
        nlu_pred_socre, nlu_ppn_pred_score = _compute_nlu_score(log, added_actions, deleted_actions)
        data = {
            "iteration_id": log["iteration_id"],
            "process_id": log["process_id"],
            "episode_id": log["episode_id"],
            "pred_recall": nlu_pred_socre["recall"],
            "pred_precision": nlu_pred_socre["precision"],
            "pred_f1": nlu_pred_socre["f1"],
            "ppn_pred_recall": nlu_ppn_pred_score["recall"],
            "ppn_pred_precision": nlu_ppn_pred_score["precision"],
            "ppn_pred_f1": nlu_ppn_pred_score["f1"]
        }
        for domain in DOMAINS:
            data[f"domain_{domain}"] = domain in log["initial_goal"]
        total.append(data)
    actions = {"added_actions": dict(sorted(Counter(added_actions).items(), key=lambda x: x[1], reverse=True)),
               "deleted_actions": dict(sorted(Counter(deleted_actions).items(), key=lambda x: x[1], reverse=True))}
    # json.dump(actions, open("actions.json", "w"), indent=4)

    df = pd.DataFrame(total).sort_values(by=["iteration_id", "process_id", "episode_id"])
    df["dialog_id"] = pd.RangeIndex(start=0, stop=len(df.index), step=1)
    df_by_domain = {domain: df[df[f"domain_{domain}"]] for domain in DOMAINS}
    return df, df_by_domain, actions

def load_main_and_nlu(log_dpath):
    df_main_score, df_main_by_domain_score = load_main_score(log_dpath)
    df_nlu_score, df_nlu_by_domain_score = load_nlu_score(log_dpath)
    df = pd.merge(df_main_score, df_nlu_score)
    df_by_domain = {domain: pd.merge(df_main_by_domain_score[domain], df_nlu_by_domain_score[domain]) for domain in DOMAINS}
    return df, df_by_domain

def score(args, save_fig=True):
    log_dpath = os.path.join(ROOT_DPATH, "outputs", args.run_type, args.run_id, args.log_part)
    if not os.path.exists(log_dpath):
        print("log_dpath: ", log_dpath)
        raise FileNotFoundError
    
    print("Evaluating {} {} {} score...".format(args.run_type, args.run_id, args.log_part))

    png_dpath = os.path.join(os.path.dirname(log_dpath), args.log_part[:-3]+"result_figures")
    if not os.path.exists(png_dpath):
        os.makedirs(png_dpath)
    keys = []
    f1_keys = []
    prec_keys = []
    recall_keys = []
    if args.score_type == "main":
        df, df_by_domain = load_main_score(log_dpath)
        keys = [
            "episode_id", "task_complete", "task_success", "book_rate", "inform_F1", "common_reward_avg", "turn",
        ]
    elif args.score_type == "nlu":
        df, df_by_domain, actions = load_nlu_score(log_dpath)
        keys = [
            "pred_recall", "pred_precision", "pred_f1", "ppn_pred_recall", "ppn_pred_precision", "ppn_pred_f1"
        ]
        f1_keys = ["pred_f1", "ppn_pred_f1"]
    elif args.score_type == "main_nlu":
        df, df_by_domain = load_main_and_nlu(log_dpath)
        keys = [
            "episode_id", "task_complete", "task_success", "book_rate", "inform_F1", "common_reward_avg", "turn",
            "pred_recall", "pred_precision", "pred_f1", "ppn_pred_recall", "ppn_pred_precision", "ppn_pred_f1"
        ]
    
    if save_fig:
        plot_sns(df=df, keys=keys, png_dpath=png_dpath)
        if f1_keys:
            plot_sns_f1s(df=df, f1_keys=f1_keys, png_dpath=png_dpath)
        # for domain, df_domain in df_by_domain.items():
        #     plot_sns(df=df_domain, keys=keys, png_dpath=os.path.join(png_dpath, domain))
    return df, df_by_domain, actions

def scores(args, run_id_list):
    df_list = []
    actions = {}
    for run_id in run_id_list:
        args_ = deepcopy(args)
        args_.run_id = run_id
        df, df_by_domain, actions = score(args=args_, save_fig=False)
        df_list.append(df)
        actions_dpath = os.path.join(ROOT_DPATH, "outputs", args.run_type, run_id, args.log_part[:-3]+"result_actions")
        if not os.path.exists(actions_dpath):
            os.makedirs(actions_dpath)
        json.dump(actions, open(os.path.join(actions_dpath, "actions.json"), "w"), indent=4)
    png_dpath = os.path.join(ROOT_DPATH, "outputs", args.run_type, run_id[:-4], args.log_part[:-3]+"result_figures")
    if not os.path.exists(png_dpath):
        os.makedirs(png_dpath)
    df = pd.concat(df_list, ignore_index=True)
    if args.score_type == "nlu":
        keys = [
            "pred_recall", "pred_precision", "pred_f1", "ppn_pred_recall", "ppn_pred_precision", "ppn_pred_f1"
        ]
        f1_keys = ["pred_f1"]# , "ppn_pred_f1"]
    else:
        raise NotImplementedError
    plot_sns_f1s(df=df, f1_keys=f1_keys, png_dpath=png_dpath)
    print(png_dpath)

if __name__ == "__main__":    
    run_id_list = [
        "bert-rule-mle-template-n-d-p--rand-d10k-bcepnum20-r12",
        "bert-rule-mle-template-n-d-p--rand-d10k-bcepnum20-r34",
        "bert-rule-mle-template-n-d-p--rand-d10k-bcepnum20-r56",
        "bert-rule-mle-template-n-d-p--rand-d10k-bcepnum20-r78",
        "bert-rule-mle-template-n-d-p--rand-d10k-bcepnum20-r90",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('run_type', choices=["rl_train", "test_single", "test_all"])
    parser.add_argument('--run_id', type=str, default="multi")
    parser.add_argument('--log_part', choices=["rl_train_log", "test_single_log", "test_all_log"])
    parser.add_argument('--score_type', choices=["main", "nlu", "main_nlu"], default="main")
    args = parser.parse_args()

    if args.run_id != "multi":
        score(args)
    elif run_id_list:
        scores(args, run_id_list)
    else:
        raise RuntimeError