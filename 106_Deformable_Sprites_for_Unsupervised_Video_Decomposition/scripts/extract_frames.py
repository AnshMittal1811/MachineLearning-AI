import argparse
import imageio
import os
import subprocess


def extract_frames(args):
    name = os.path.splitext(os.path.basename(args.path))[0]

    r = imageio.get_reader(args.path)
    meta = r.get_meta_data()
    duration = meta["duration"]
    start = args.start if args.start >= 0 else start + duration
    end = args.end if args.end >= 0 else args.end + duration

    outdir = "{}/{}_{}-{}_fps{}".format(args.outdir, name, start, end, args.fps)
    if not args.overwrite and os.path.isdir(outdir):
        print(f"{outdir} already exists, exiting.")
        return

    os.makedirs(outdir, exist_ok=True)

    scale_str = ""
    if args.width > 0:
        scale_str = f",scale={args.width}:-1"
    elif args.height > 0:
        scale_str = f",scale=-1:{args.height}"

    arg_str = f"-copyts -vf fps={args.fps}{scale_str}"
    cmd = f"ffmpeg -ss {start} -i {args.path} -to {end} {arg_str} {outdir}/%05d.{args.ext} -n"
    cmd = f"{cmd} -loglevel error"
    print(cmd)

    subprocess.call(cmd, shell=True, stdin=subprocess.PIPE)
    if not os.path.isdir(outdir):
        print("Could not extract frames to {}".format(outdir))
        return
    print("extracted frames to", outdir)

    print("making gif for rgb frames in", outdir)
    outpath = os.path.join(outdir, "frames.gif")
    gif_cmd = f"ffmpeg -f image2 -framerate {args.fps} -i {outdir}/%05d.{args.ext} {outpath} -n"
    gif_cmd = f"{gif_cmd} -loglevel error"
    print(gif_cmd)
    subprocess.call(gif_cmd, shell=True, stdin=subprocess.PIPE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to video")
    parser.add_argument("outdir", type=str, help="output dir for frames")
    parser.add_argument(
        "--ext", type=str, default="png", help="output filetype for frames"
    )
    parser.add_argument(
        "--width", type=int, default=480, help="target width for frames"
    )
    parser.add_argument(
        "--height", type=int, default=-1, help="target height for frames"
    )
    parser.add_argument("--fps", type=int, default=10, help="fps to extract frames")
    parser.add_argument("--start", type=float, default=0, help="seconds to start")
    parser.add_argument("--end", type=float, default=-1, help="seconds to end")
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite if already exist"
    )

    args = parser.parse_args()
    extract_frames(args)
