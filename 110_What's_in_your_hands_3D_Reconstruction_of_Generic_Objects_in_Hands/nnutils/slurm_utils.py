import subprocess
import os
from typing import Any
import submitit

def wrap_cmd(cmd):
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    try:
        print('wait')
        p.wait()
    except KeyboardInterrupt:
        try:
            print('Detect Ctrl+C, terminating....')
            p.terminate()
        except OSError:
            pass
        p.wait()


def slurm_wrapper_hydra(args, callable_fn):
    """name matching for hydra environment"""
    if args.environment.slurm:
        executor = submitit.AutoExecutor(
            folder=args.logging.submitit_dir,
            slurm_max_num_timeout=100,
            cluster=None if args.environment.slurm else "debug",
        )
        # asks SLURM to send USR1 signal 30 seconds before the time limit
        # additional_parameters = {"signal": 'USR1@120'}
        additional_parameters = {"signal": 'SIGUSR1@90'}
        if args.environment.nodelist != "":
            additional_parameters = {"nodelist": args.environment.nodelist}
        if args.environment.exclude_nodes != "":
            additional_parameters.update(
                {"exclude": args.environment.exclude_nodes.replace('+', ',')})
        args.environment.workers = 10 * args.environment.ngpu
        executor.update_parameters(
            timeout_min=args.environment.slurm_timeout,
            slurm_partition=args.environment.slurm_partition,
            cpus_per_task=args.environment.workers,
            gpus_per_node=args.environment.ngpu,
            nodes=args.environment.world_size,
            tasks_per_node=1,
            mem_gb=args.environment.mem_gb,
            slurm_additional_parameters=additional_parameters,
            signal_delay_s=120)
        executor.update_parameters(name=args.expname+args.logging.suffix)
        job = executor.submit(callable_fn, args)
        if not args.environment.slurm:
            job.result()
    else:
        callable_fn(args)
    return 



class Worker:
    def checkpoint(self, *args: Any, **kwargs: Any):
        # for resubmit
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)  # submits to requeuing    

    def __call__(self, func, **kwargs):
        func(**kwargs)


def slurm_wrapper(args, save_dir, func, func_kwargs, resubmit=True):
    """name matching for my own slurm args"""
    job_name = '/'.join(save_dir.split('/')[-2:])
    save_dir = save_dir  + '/submitit_cache'
    if args.slurm:
        # asks SLURM to send USR1 signal 30 seconds before the time limit
        additional_parameters = {"signal": 'SIGUSR1@90', }
        executor = submitit.AutoExecutor(folder=save_dir)
        executor.update_parameters(
            tasks_per_node=args.sl_ntask_pnode,
            timeout_min=args.sl_time,
            slurm_partition=args.sl_part,
            gpus_per_node=args.sl_ngpu,
            cpus_per_task=args.sl_ngpu * 10,
            nodes=args.sl_node,
            slurm_job_name=job_name,
            slurm_additional_parameters=additional_parameters,
            slurm_signal_delay_s=120
        )
        print('slurm cache in ', save_dir)
        executor.update_parameters(
            name=job_name
        )
        with executor.batch():
            for _ in range(args.sl_node):
                if resubmit:
                    job = executor.submit(Worker(), {'func': func, 'kwargs': func_kwargs})
                else:
                    job = executor.submit(func, **func_kwargs)

    else:
        func(**func_kwargs)
        job = None
    return job

    
def add_slurm_args(arg_parser):
    arg_parser.add_argument(
        "--slurm",
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument("--sl_time",default=1080, type=int)  # 16 hrs
    arg_parser.add_argument("--sl_dir", default='/checkpoint/yufeiy2/slurm_cache_shot', type=str)  
    arg_parser.add_argument("--sl_work",default=10, type=int)
    arg_parser.add_argument("--sl_node",default=1, type=int)  # 16 hrs
    arg_parser.add_argument("--sl_ngpu",default=8, type=int)
    arg_parser.add_argument("--sl_ntask_pnode",default=1, type=int)
    arg_parser.add_argument("--sl_part",default='learnlab', type=str)
    return arg_parser


