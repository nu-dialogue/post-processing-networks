# NLU
from nlu.joint_bert import JointBertNLU, JointBertNLUPPN
from nlu.svm import SVMNLU, SVMNLUPPN

# DST
from dst.rule import RuleDST, RuleDSTPPN
from dst.trade import TRADEDST, TRADEDSTPPN

# Policy
from policy.rule import RulePolicy, RulePolicyPPN
from policy.mle import MLEPolicy, MLEPolicyPPN
from policy.ppo import PPOPolicy, PPOPolicyPPN
from policy.larl import LaRLPolicy
from policy.hdsa import HDSAPolicy

# NLG
from nlg.template import TemplateNLG, TemplateNLGPPN
from nlg.sclstm import SCLSTMNLG, SCLSTMNLGPPN

MODULE_DICT = {
    "nlu": {
        "bert": {
            "module": JointBertNLU,
            "ppn": JointBertNLUPPN
        },
        "svm": {
            "module": SVMNLU,
            "ppn": SVMNLUPPN
        }
    },
    "dst": {
        "rule": {
            "module": RuleDST,
            "ppn": RuleDSTPPN
        },
        "trade": {
            "module": TRADEDST,
            "ppn": TRADEDSTPPN
        }
    },
    "policy": {
        "rule": {
            "module": RulePolicy,
            "ppn": RulePolicyPPN,
        },
        "mle": {
            "module": MLEPolicy,
            "ppn": MLEPolicyPPN
        },
        "ppo": {
            "module": PPOPolicy,
            "ppn": PPOPolicyPPN
        },
        "larl": {
            "module": LaRLPolicy
        },
        "hdsa": {
            "module": HDSAPolicy
        }
    },
    "nlg": {
        "template": {
            "module": TemplateNLG,
            "ppn": TemplateNLGPPN
        },
        "sclstm": {
            "module": SCLSTMNLG,
            "ppn": SCLSTMNLGPPN
        }
    }
}

WORD_DST_LIST = [
    "trade"
]

WORD_POLICY_LIST = [
    "larl",
    "hdsa"
]