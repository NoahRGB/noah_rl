# NoahRGB 10/2026
#
# a script that wraps some logging code so that all
# configs/networks are saved in the same place
#
# e.g.
# python run.py <algorithm> [title=...] [notes="..."] [hydra overrides]
# python run.py ppo title=shooting notes="testing if ppo can shoot" lr=0.001 cgn=0.5
#
# title defaults to the algorithm name, and the run is saved to
# results/logs/<title>_<YYYYMMDD>_<HHMMSS>
#

import os, re, sys, time, shlex, importlib, subprocess
from datetime import datetime

import hydra
from omegaconf import DictConfig

from noah_rl.utils.logging import update_summary

ROOT = os.path.dirname(os.path.abspath(__file__))

def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def write_notes(run_dir: str, algo: str, notes: str, command: str):
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    status = git("status", "--porcelain")
    git_line = commit
    if status:
        # git diff doesn't include untracked files, so list them separately
        diff = git("diff", "HEAD")
        untracked = [line[3:] for line in status.splitlines() if line.startswith("??")]
        with open(os.path.join(run_dir, "code.diff"), "w") as f:
            f.write(diff or "")
            if untracked:
                f.write("\n# untracked files (not included above):\n")
                f.writelines(f"#   {path}\n" for path in untracked)
        git_line += " (uncommitted changes, see code.diff)"

    # write notes.md with algortihm, date/timee, notes,
    # command, git details
    with open(os.path.join(run_dir, "notes.md"), "w") as f:
        f.write(f"# {algo} - {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"{notes or '(no notes)'}\n\n")
        f.write(f"command: `{command}`\n\n")
        f.write(f"git: {git_line}\n\n")
        f.write("## Results\n\n")

def pop_arg(argv: list, name: str):
    # removes the given name from the list
    for i, arg in enumerate(argv):
        if arg.startswith(f"{name}="):
            argv.pop(i)
            return arg[len(f"{name}="):]
    return None

command = shlex.join(["python", *sys.argv])
algo = None
notes = None

@hydra.main(config_path="params", version_base=None)
def main(cfg: DictConfig):
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    write_notes(run_dir, algo, notes, command)

    start = time.time()
    update_summary(run_dir, algo=algo, title=cfg.get("title"), status="running",
                   started_at=datetime.now().isoformat(timespec="seconds"))

    def finish(status: str, **fields):
        update_summary(run_dir, status=status, ended_at=datetime.now().isoformat(timespec="seconds"),
                       duration_s=round(time.time() - start), **fields)

    # run the training loop, catch early quits
    try:
        importlib.import_module(f"noah_rl.algorithms.{algo}").train(cfg, run_dir)
    except KeyboardInterrupt:
        finish("interrupted")
        raise
    except Exception as e:
        finish("crashed", error=f"{type(e).__name__}: {e}")
        raise
    finish("finished")

if __name__ == "__main__":

    # check the command lists an algorithm to run
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit("usage: python run.py <algorithm> [title=...] [notes=\"...\"] [hydra overrides...]")

    # check that the algortih exists
    algo = sys.argv.pop(1)
    if not os.path.isfile(os.path.join(ROOT, "params", f"{algo}.yaml")):
        sys.exit(f"no config found at params/{algo}.yaml")

    # remove title arg sinc hydra doesnt need to see it
    # repeatedly prompt for title if it isnt given and use
    # valid characters
    title = pop_arg(sys.argv, "title")
    if title is None and sys.stdin.isatty():
        while True:
            title = input(f"title for this run (enter for '{algo}'): ").strip() or algo
            if re.fullmatch(r"[A-Za-z0-9_.-]+", title):
                break
            print("title can only contain letters, numbers, _ . -")
    title = title or algo
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", title):
        sys.exit(f"title '{title}' can only contain letters, numbers, _ . -")

    # remove teh notes arg since hydra doesn't need to see it
    # prompt for notes if none are given
    notes = pop_arg(sys.argv, "notes")
    if notes is None and sys.stdin.isatty():
        notes = input("notes for this run (enter to skip): ").strip()

    sys.argv.insert(1, f"--config-name={algo}")

    # stop hydra creating a .log file (annoying)
    sys.argv.insert(2, "hydra/job_logging=none")

    # add title/rundir to the yaml configs
    sys.argv.insert(3, f"++title={title}")
    sys.argv.insert(4, f"hydra.run.dir=results/logs/{title}_{datetime.now():%Y%m%d_%H%M%S}")
    main()
