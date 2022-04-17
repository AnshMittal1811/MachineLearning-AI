import argparse
import glob
import os
import json
import numpy as np


def gather_metric(run_dir, name="mean_ious"):
    """
    gather all instances of metric `name` computed during the run sorted by mtime
    :returns list of the metric `name`
    """
    metrics = {}
    for root, subds, files in os.walk(run_dir):
        if f"{name}.txt" in files:
            path = os.path.join(root, f"{name}.txt")
            metrics[path] = np.loadtxt(path).tolist()
    if len(metrics) < 1:
        return []
    # sort the dict by last modified time
    mod_sorted = sorted(metrics.keys(), key=os.path.getmtime)
    return [metrics[k] for k in mod_sorted]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root")
    parser.add_argument(
        "--run",
        type=int,
        default=-1,
        help="which run (relative to all existing) to select [default -1]",
    )
    parser.add_argument("--metric", default="mean_ious")
    args = parser.parse_args()

    ## gather runs for jobs
    job_runs = {}
    for root, subds, files in os.walk(args.log_root):
        if ".hydra" in subds:
            job_dir = os.path.dirname(root)
            if job_dir not in job_runs:
                job_runs[job_dir] = []
            job_runs[job_dir].append(root)

    # select runs for each job
    sel_runs = {job: runs[args.run] for job, runs in job_runs.items()}

    # get all the eval outputs for each run
    latest = {}
    best = {}
    i_best = {}
    for job, run_dir in sel_runs.items():
        vals = gather_metric(run_dir, args.metric)
        if len(vals) < 1:
            print("metric {} computed 0 times in {}".format(args.metric, run_dir))
            import ipdb

            ipdb.set_trace()
        latest[job] = vals[-1]
        best[job] = np.max(vals, axis=0).tolist()
        i_best[job] = np.argmax(vals, axis=0).tolist()

    out_dict = {"latest": latest, "best": best, "i_best": i_best}
    out_dict["avg_latest"] = np.mean(np.stack(latest.values(), axis=0), axis=0).tolist()
    out_dict["avg_best"] = np.mean(np.stack(best.values(), axis=0), axis=0).tolist()

    out_path = os.path.join(args.log_root, "{}_run{}".format(args.metric, args.run))
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=1)

    print(
        "aggregated all {} for {} runs in {}".format(
            args.metric, len(sel_runs), out_path
        )
    )
