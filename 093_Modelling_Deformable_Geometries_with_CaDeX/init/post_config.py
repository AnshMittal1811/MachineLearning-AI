import os
import sys
import shutil
import platform
from pprint import pformat
import yaml
import time
import logging
from datetime import datetime
from .config_utils import configure_logging


def post_config(cfg, interactive=True):
    """
    preparation before start
    """
    local_time = time.asctime(time.localtime(time.time()))
    local_time = "_".join(local_time.split(" "))

    # check init
    os_cate = platform.system().lower()
    assert os_cate in ["linux"], "Only support Linux now!"

    print("=" * shutil.get_terminal_size()[0])
    print("Please check the configuration")
    print("-" * shutil.get_terminal_size()[0])
    print(pformat(cfg))
    print("-" * shutil.get_terminal_size()[0])
    print("y/n?", end="")
    if interactive:
        need_input = True
        while need_input:
            response = input().lower()
            if response in ("y", "n"):
                need_input = False
        if response == "n":
            sys.exit()
    else:
        print("y Warning, NO INTERACTIVE CONFIRM!")
    print("=" * shutil.get_terminal_size()[0])

    # check log dir
    if cfg["resume"] == "None":
        cfg["resume"] = False
    abs_log_dir = os.path.join(cfg["root"], "log", cfg["logging"]["log_dir"])
    if cfg["resume"]:
        # if resume, don't remove the old log
        if not os.path.exists(abs_log_dir):
            print(
                "Warning: Need resume but the log dir: "
                + abs_log_dir
                + " doesn't exist, create new log dir and not resume",
                end="",
            )
            os.makedirs(abs_log_dir)
            cfg["resume"] = False
        else:
            print("resume from {}, log dir found".format(cfg["resume"]))
    else:
        if os.path.exists(abs_log_dir):
            print(
                "Warning! No resume but log dir: "
                + abs_log_dir
                + " exists. Remove the old dir? y/n"
            )
            if interactive:
                need_input = True
                while need_input:
                    response = input().lower()
                    if response in ("y", "n"):
                        need_input = False
                if response == "n":
                    sys.exit()
            else:
                print("y Warning, NO INTERACTIVE CONFIRMATION, RENAME OLD LOG DIR!")
            # os.system("rm " + abs_log_dir + " -r")
            os.makedirs(abs_log_dir + "_old", exist_ok=True)
            os.system(
                f"mv {abs_log_dir} "
                + os.path.join(
                    abs_log_dir + "_old",
                    os.path.basename(abs_log_dir)
                    + f"_dup_old_rename_at_{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}",
                )
            )
        os.makedirs(abs_log_dir)
    print("Log dir confirmed...")

    configure_logging(cfg, time_stamp=local_time)

    # prepare dataset proportion
    if isinstance(cfg["dataset"]["dataset_proportion"], float):
        cfg["dataset"]["dataset_proportion"] = [cfg["dataset"]["dataset_proportion"]] * len(
            cfg["modes"]
        )

    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])
    logging.info("Set GPU: " + str(cfg["gpu"]) + " ...")

    # backup model, dataset, init and running init
    file_backup_path = os.path.join(abs_log_dir, "files_backup")
    os.makedirs(file_backup_path, exist_ok=True)

    logging.info("Save configuration to local file...")
    with open(
        os.path.join(file_backup_path, "running_config_{}.yaml".format(local_time)), "w+"
    ) as f:
        f.write(yaml.dump(cfg))

    backup_fn_list = [
        os.path.join(cfg["root"], "core", "models", cfg["model"]["model_name"] + ".py"),
        os.path.join(cfg["root"], "dataset", cfg["dataset"]["dataset_name"] + ".py"),
    ] + [os.path.join(cfg["root"], filename) for filename in cfg["logging"]["backup_files"]]
    for fn in backup_fn_list:
        try:
            shutil.copy2(fn, file_backup_path)
        except:
            logging.warning("{} backup failed".format(fn))

    return cfg
