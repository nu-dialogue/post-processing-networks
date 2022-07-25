import os
import json
from copy import deepcopy
from util import plot_sns, DOMAINS, get_logger
logger = get_logger(__name__)
from system import SYSTEM

def load_system_config(self, resume_rl_train_dpath):
    base_system_config = {}
    for module_type, module_name in self.module_list.items():
        if not module_name:
            # If the module was not used
            base_system_config[module_type] = {"module_name": ""}
        else:
            ppn_type = f"{module_type}_ppn"
            ppn_config_fpath = os.path.join(resume_rl_train_dpath, "ppn_configs", ppn_type+".json")
            if not os.path.exists( ppn_config_fpath ):
                # If ppn was not used
                ppn_config = {"use": False}
            else:
                # If ppn was used
                ppn_config = json.load(open(ppn_config_fpath))
                # assert ppn_config["module_name"] == module_name
                ppn_config["ppn_type"] = ppn_type
            base_system_config[module_type] = {"module_name": module_name, "ppn_config": ppn_config}
    return base_system_config

def make_system(base_system_config, ppn_config_dpath, resume_rl_train_id, iteration_id):
    logger.info("Making System...")
    system_config = deepcopy(base_system_config)
    for module_type, module_config in system_config.items():
        ppn_type = f"{module_type}_ppn"
        if not module_config["module_name"]:
            continue
        if not module_config["ppn_config"]["use"]:
            continue
        module_config["ppn_config"]["ppn_config_fpath"] = os.path.join(ppn_config_dpath, ppn_type+".json")
        module_config["ppn_config"]["resume_rl_train_id"] = resume_rl_train_id
        module_config["ppn_config"]["resume_rl_iteration_id"] = iteration_id
        module_config["ppn_config"]["resume_rl_trained_model_fpath"] = os.path.join(module_config["ppn_config"]["trained_model_dpath"], 
                                                                                    f"{iteration_id}.model")
        module_config["ppn_config"]["trained_model_dpath"] = ""
    return SYSTEM(system_config=system_config)


def save_table(iteration_df, result_table_dpath, iteration_id):
    if result_table_dpath is None:
        logger.info("Specify result_table_dapth.")
        return
    iteration_df.to_csv(os.path.join(result_table_dpath, f"raw_{iteration_id}.csv"))
    iteration_df.mean(numeric_only=True).to_csv(os.path.join(result_table_dpath, f"mean_{iteration_id}.csv"))
    logger.info("Save result table to {}".format( os.path.join(result_table_dpath, f"mean_{iteration_id}.csv") ))
    # for domain in DOMAINS:
    #     domain_results_dpath = os.path.join(self.results_dpath, domain)
    #     if not os.path.exists(domain_results_dpath):
    #         os.makedirs(domain_results_dpath)
    #     domain_df = total_df[total_df[f"{domain}_in"]]
    #     domain_df.to_csv(os.path.join(domain_results_dpath, "raw.csv"))
    #     domain_df.mean().to_csv(os.path.join(domain_results_dpath, "mean.csv"))

def print_score(iteration_df, metrics):
    log_str = "\t\t".join(["{}: {:.5f}".format(key, score) for key, score in iteration_df[metrics].mean().items()])
    logger.info(log_str)

def plot(total_df, metrics, result_figure_dpath):
    df_by_domain = {domain: total_df[total_df[f"domain_{domain}"]] for domain in DOMAINS}

    plot_sns(df=total_df, keys=metrics, png_dpath=result_figure_dpath)
    for domain, df_domain in df_by_domain.items():
        plot_sns(df=df_domain, keys=metrics, png_dpath=os.path.join(result_figure_dpath, domain))
