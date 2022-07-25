import os
import pandas as pd
import random

id_list = [
	"bert-rule-rule-sclstm-n-d-p--rand-d10k-bcepnum20-r17895",
	"bert-rule-rule-sclstm-n-d-p--rand-d10k-bcepnum20-r36214",
	"bert-rule-rule-sclstm-n-d-p--rand-d10k-bcepnum20-r40516",
	"bert-rule-rule-sclstm-n-d-p--rand-d10k-bcepnum20-r48403",
	"bert-rule-rule-sclstm-n-d-p--rand-d10k-bcepnum20-r55961",
	"bert-rule-rule-sclstm-n-d-p--rand-d10k-bcepnum20-r65759",
	"bert-rule-rule-sclstm-n-d-p--rand-d10k-bcepnum20-r66221",
	"bert-rule-rule-sclstm-n-d-p--rand-d10k-bcepnum20-r75737",
	"bert-rule-rule-sclstm-n-d-p--rand-d10k-bcepnum20-r80927",
	"bert-rule-rule-sclstm-n-d-p--rand-d10k-bcepnum20-r91854"
]
id_list = [id_list[i] for i in random.sample(list(range(10)), k=5)]
test_single_dpath = "outputs/test_single"

df_list = []
for id_ in id_list:
    raw_fpath = os.path.join(test_single_dpath, id_, "test_single_result_tables", "raw_.csv")
    df_list.append(pd.read_csv(raw_fpath, index_col=0))

total_df = pd.concat(df_list, ignore_index=True)
print(id_list[0][:-7])
for r in sorted([id_[-5:] for id_ in id_list]):
	print(r)
print(total_df.mean(numeric_only=True))