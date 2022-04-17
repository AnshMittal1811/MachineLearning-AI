import argparse
import os
import subprocess

SRC_DIR = "/home/vye/multi-style-transfer/experiments"
STYLE_DIR = os.path.join(SRC_DIR, "images/21styles")
model_path = os.path.join(SRC_DIR, "models/21styles.model")

STYLES = sorted(os.listdir(STYLE_DIR))

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--style", choices=STYLES, default="candy.jpg")
parser.add_argument("--size", type=int, default=240)
args = parser.parse_args()

base_dir = os.path.dirname(args.path)
src_file = os.path.basename(args.path)
style_name = os.path.splitext(args.style)[0]
out_path = os.path.join(base_dir, f"edited_{style_name}_{src_file}")

exe = os.path.join(SRC_DIR, "main.py")

base_cmd = "python {} eval --model {}".format(exe, model_path)
cmd = "{} --style-image {}/{} --content-image {} --content-size {} --output-image {}".format(
    base_cmd, STYLE_DIR, args.style, args.path, args.size, out_path
)
print(cmd)
subprocess.call(cmd, shell=True)
