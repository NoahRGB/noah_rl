# NoahRGB 10/2026
#
# a script to watch an agent from a training run play,
# using the exact config that the run was trained with
#
# e.g.
# python watch.py <run_dir> [overrides]
# python watch.py results/logs/ppo_pixels_20261020_100129
# python watch.py ppo_pixels_20261020_100129 num_envs=2
#

import os, sys, json, tempfile, importlib

from omegaconf import OmegaConf
from torch.optim.optimizer import register_optimizer_step_pre_hook

ROOT = os.path.dirname(os.path.abspath(__file__))

# common config options when doing a watch/eval
WATCH_DEFAULTS = {
    "num_envs": 1,
    "save_network": False,
    "env_kwargs": {"render_mode": "human"},
}

def find_run_dir(arg: str):
    # look for the specified run directory as a path or in 
    # results/logs/ and exit if not found
    for path in [arg, os.path.join(ROOT, "results", "logs", arg)]:
        if os.path.isdir(path):
            return path
    sys.exit(f"no run found at {arg} or results/logs/{arg}")

def freeze_optimisers():
    # set lr=0 for every parameter to stop the algorithm learning
    # durin an eval/watch
    def zero_lr(optim, args, kwargs):
        for group in optim.param_groups:
            group["lr"] = 0.0
    register_optimizer_step_pre_hook(zero_lr)

def build_config(run_dir: str, algo: str, overrides: list):
    # find thee actual hydra config in results/logs/ and use that for the eval/watch
    # which will replicate the training config exactly
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))

    # get any extra "eval" config params specified in the .yaml
    algo_eval = OmegaConf.load(os.path.join(ROOT, "params", f"{algo}.yaml")).get("eval") or {}
    cfg = OmegaConf.merge(cfg, WATCH_DEFAULTS, algo_eval)

    cfg.pop("eval", None)
    cfg.load_path = os.path.join(run_dir, "network.pt")
    cfg.title = f"watch_{cfg.get('title', algo)}"

    OmegaConf.set_struct(cfg, True)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

if __name__ == "__main__":

    # make sure a run dir is actually specified
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit("usage: python watch.py <run_dir> [overrides...]")

    # make sure the run dir exists
    run_dir = find_run_dir(sys.argv[1])
    if not os.path.isfile(os.path.join(run_dir, "network.pt")):
        sys.exit(f"no network.pt in {run_dir} (was it trained with save_network=True?)")

    # make sure the summary.json exists in the run dir and find the alg name
    try:
        with open(os.path.join(run_dir, "summary.json")) as f:
            algo = json.load(f)["algo"]
    except (OSError, KeyError):
        sys.exit(f"couldn't read the algorithm name from {run_dir}/summary.json")

    # set the eval config , stop learning 
    cfg = build_config(run_dir, algo, sys.argv[2:])
    freeze_optimisers()
    print(f"watching {algo} from {cfg.load_path}")

    # make a temporary log dir for the eval run so nothing complains
    log_dir = tempfile.mkdtemp(prefix="noah_rl_watch_")
    try:
        # run the actual eval 
        importlib.import_module(f"noah_rl.algorithms.{algo}").train(cfg, log_dir)
    except KeyboardInterrupt:
        pass
