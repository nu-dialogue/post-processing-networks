from modules import (
    MODULE_DICT,
    WORD_DST_LIST,
    WORD_POLICY_LIST
)

def create_module(module_type, system_config):
    module_config = system_config[module_type]
    module_name = module_config["module_name"]
    if not module_config["module_name"]:
        if module_type == "nlu" and system_config["dst"]["module_name"] in WORD_DST_LIST:
            return None
        elif module_type == "nlg" and system_config["policy"]["module_name"] in WORD_POLICY_LIST:
            return None
        else:
            raise Exception("{} can not be None.".format(module_type))
    try:
        return MODULE_DICT[module_type][module_name]["module"](module_type=module_type,
                                                               module_config=module_config)
    except KeyError as e:
        raise KeyError("{}'s {} is not defined.".format(module_type, module_name))
    except Exception as e:
        raise Exception(module_type, module_name)

def create_ppn(module, system_state_dim):
    if not module:
        return None
    elif not module.ppn_config["use"]:
        return None
    try:
        return MODULE_DICT[module.type][module.name]["ppn"](module=module,
                                                            system_state_dim=system_state_dim)
    except KeyError as e:
        raise KeyError("{} {}'s ppn is not defined.".format(module.type, module.name))