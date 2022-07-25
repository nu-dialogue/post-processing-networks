import sys
import os
from glob import glob
from tqdm import tqdm
import pandas as pd

ROOT_DPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DPATH)
OUTPUT_DPATH = os.path.join(ROOT_DPATH, "outputs")

from util import sub_plot_sns, multi_plot_sns

def make_result_tables_dpaths(base_rl_train_id, random_seed_list):
    dpaths = []
    for random_seed in random_seed_list:
        dpaths.append(os.path.join(OUTPUT_DPATH,
                                   "rl_train",
                                   f"{base_rl_train_id}-r{random_seed}",
                                   "rl_train_result_tables"))
    return dpaths

def load_raw_result_tables(result_tables_dpaths):
    df_list = []
    for result_tables_dpath in result_tables_dpaths:
        print(f"Loading result tables from {result_tables_dpath} .")
        for result_table_fpath in tqdm(glob(os.path.join(result_tables_dpath, "raw_*.csv"))):
            df_list.append(pd.read_csv(result_table_fpath, index_col=0))
    return pd.concat(df_list, ignore_index=True)

    
def main(base_rl_train_ids, keys, merge_id):
    df_list = []
    for strategy, (base_rl_train_id, random_seed_list) in base_rl_train_ids.items():
        dpaths = make_result_tables_dpaths(base_rl_train_id=base_rl_train_id,
                                           random_seed_list=random_seed_list)
        df = load_raw_result_tables(result_tables_dpaths=dpaths)
        df_list.append(df.assign(strategy=strategy))

    png_dpath = os.path.join(OUTPUT_DPATH,
                             "rl_train",
                             merge_id,
                             "rl_train_result_figures")
    # sub_plot_sns(df=pd.concat(df_list, ignore_index=True), keys=["task_success", "inform_F1", "turn"], png_dpath=png_dpath)
    multi_plot_sns(df=pd.concat(df_list, ignore_index=True), keys=keys, png_dpath=png_dpath)

if __name__ == "__main__":
    base_rl_train_ids = {
        # "ALL": ["bert-rule-mle-template-n-d-p--simul-d10k-bcepnum20", [12, 34 ,56, 78, 90]],
        # "RANDOM": ["bert-rule-mle-template-n-d-p--rand-d10k-bcepnum20", [12, 34 ,56, 78, 90]],
        # "ROTATION": ["bert-rule-mle-template-n-d-p--rot-d10k-bcepnum20", [12, 34 ,56, 78, 90]],
        "MLE": ["bert-rule-mle-template-n-d-p--rand-d10k-bcepnum20", [12, 34 ,56, 78, 90]],
        "PPO": ["bert-rule-ppo-template-n-d-p--rand-d10k-bcepnum20", [12636, 12764, 53744, 89251, 95529]],
        "SCLSTM":  ["bert-rule-rule-sclstm-n-d-p--rand-d10k-bcepnum20", [17895, 36214, 40516, 48403, 55961]],
        "TRADE": ["trade-rule-template--d-p--rand-d10k-bcepnum20", [28448, 29680, 36678, 60153, 77780]],
        "LaRL": ["bert-rule-larl--n-d---rand-d10k-bcepnum20", [17960, 23148, 43722, 61101, 84506, 93066]]
    }
    merge_id = "all"
    keys = ["task_success", "book_rate", "inform_F1", "inform_precision", "inform_recall", "turn"]

    main(base_rl_train_ids=base_rl_train_ids, keys=keys, merge_id=merge_id)
