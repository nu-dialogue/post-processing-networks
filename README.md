# Post-processing Networks
This is the implementation of SIGdial 2022 paper:

Post-processing Networks: Method for Optimizing Pipeline Task-oriented Dialogue Systems using Reinforcement Learning. [[arXiv](https://example.com)]

## Setup
Python == 3.7
1. Clone repository
    ```bash
    $ git clone --recursive git@github.com:ohashi56225/post-processing-networks.git
    $ cd post-processing-networks
    ```
2. Install ConvLab-2
    ```bash
    $ cd ConvLab-2/ && pip install -e . && cd ../
    $ python -m spacy download en_core_web_sm
    ```
3. Additional requirements

    **Please fix the pytorch's version in `requirements.txt` appropriate for your environment**
    ```bash
    $ pip install -r requirements.txt
    ```
    > It is ok to ignore pip's dependency error with ConvLab-2.

## Experiments
Instructions for training and evaluating PPNs. As an example, a system consisting of BERT NLU, Rule DST, MLE Policy, and Template NLG is used here. See `module.py` for available modules.

### 0. Module preparation
Pre-trained models available on convlab2 are used for the modules that comprise the system.
- Most models are automatically downloaded as needed, some must be downloaded manually.
    - If you use MLE Policy, see `policy/mle/README.md`
    - If you use PPO Policy, see `policy/ppo/README.md`
- In addition, the convlab2 source codes for some modules need to be modified to avoid errors and warnings.
    - If you use SVM NLU, see `nlu/svm/README.md`
    - If you use LaRL Policy, see `policy/larl/README.md`
    - If you use SCLSTM NLG, see `nlg/sclstm/README.md`

### 1. Data preparation for imiation learning
A PPN is pre-trained by imitation learning (Behavior Cloning; BC), in which the output of each module is labeled. Therefore, we generate the data for BC by actually running the dialogue simulation and sampling the output of each module.
```bash
$ python main.py \
    --run_type bcd_generate \
    --run_id 2022-0707-bert-rule-mle-template \
    --process_num 16 \
    --nlu_name bert \
    --nlu_ppn_use True \
    --dst_name rule \
    --dst_ppn_use True \
    --policy_name mle \
    --policy_ppn_use True \
    --nlg_name template
```
Finally, the sampled data for BC and dialogue logs are saved in directory `outputs/bcd_generate/2022-0707-bert-rule-mle-template`.

### 2. Reinforcement learning
Now, we perform reinforcement learning using the same combination of modules used to prepare the BC data. Also, specify the run_id of the BC data as `bcd_generate_id`.
```bash
python main.py \
    --run_type rl_train \
    --run_id 2022-0707-bert-rule-mle-template \
    --process_num 8 \
    --nlu_name bert \
    --nlu_ppn_use True \
    --dst_name rule \
    --dst_ppn_use True \
    --policy_name mle \
    --policy_ppn_use True \
    --nlg_name template \
    --rl_train_selection_strategy random \
    --bcd_generate_id 2022-0701-bert-rule-mle-template 
```
Reinforcement learning is performed for a specified number of timesteps after imitation learning with BC data is performed
Finally, models of trained PPNs, history of dialogue simulations during training, and figures of training curves are saved in directory `outputs/rl_train/2022-0707-bert-rule-mle-template`.

### 3. Evaluation
We evaluate the trained PPN models using test dialogue simulations. Specify the run_id and the iteration id you wish to test.
```bash
rl_train_id="2022-0707-bert-rule-mle-template"
rl_train_iter_id="34"
python main.py \
    --run_type test_single \
    --run_id 2022-0707-bert-rule-mle-template \
    --process_num 5 \
    --nlu_name bert \
    --nlu_ppn_use True \
    --nlu_ppn_resume_rl_train_id ${rl_train_id} \
    --nlu_ppn_resume_rl_iteration_id ${rl_train_iter_id} \
    --dst_name rule \
    --dst_ppn_use True \
    --dst_ppn_resume_rl_train_id ${rl_train_id} \
    --dst_ppn_resume_rl_iteration_id ${rl_train_iter_id} \
    --policy_name mle \
    --policy_ppn_use True \
    --policy_ppn_resume_rl_train_id ${rl_train_id} \
    --policy_ppn_resume_rl_iteration_id ${rl_train_iter_id} \
    --nlg_name template 
```
The result and dialogue histories in the evaluation are saved in directory `outputs/test_single/2022-0707-bert-rule-mle-template`. See `test_single_results_tables/mean_.csv` for summary scores of the sytem's performance.
