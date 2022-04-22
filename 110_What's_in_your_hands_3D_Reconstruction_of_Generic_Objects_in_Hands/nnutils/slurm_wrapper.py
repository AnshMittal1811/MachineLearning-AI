import argparse
import submitit 
from typing import Any

from . import slurm_utils

class Worker:
    # for resubmit
    def checkpoint(self, *args: Any, **kwargs: Any) -> submitit.helpers.DelayedSubmission:
        print('requeue!!!?????')
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)  # submits to requeuing    

    def __call__(self, cmd) -> Any:
        slurm_utils.wrap_cmd(cmd)


def submit_to_slurm(args, unknown):
    """wrap slurm"""
    # submitit --> launch
    job = slurm_utils.slurm_wrapper(
        args, 
        args.sl_dir, 
        Worker(), 
        {'cmd': ' '.join(unknown)}
    )
    return job

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    slurm_utils.add_slurm_args(parser)
    args, unknown = parser.parse_known_args()
    print("Command Line Args:", args)
    # add slurm?? 

    submit_to_slurm(args, unknown)