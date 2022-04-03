
"""Convert tfrecords to gziped files.

This file converts tfrecords in deepmind gqn dataset to gzip files. Each
tfrecord will be converted to multiple gzip files which contains a list of
tuples `(images, poses)`.

For example, when converting `shepard_metzler_5_parts` dataset with batch size
of `100`, a single tfrecord file which contains `2000` sequences is converted
to `20` gzip files which contain a list of `100` tuples.

ex) 900-of-900.tfrecord -> 900-of-900-1.pt.gz, ..., 900-of-900-20.pt.gz

Images size: `(sequence, height, width, channel)`.
Viewpoints size: `(sequence, v_dim)`

(Dataset)

https://github.com/deepmind/gqn-datasets

(Refrence)

https://github.com/deepmind/gqn-datasets/blob/master/data_reader.py

https://github.com/l3robot/gqn_datasets_translator

https://github.com/wohlert/generative-query-network-pytorch/blob/master/scripts/tfrecord-converter.py

https://github.com/iShohei220/torch-gqn/blob/master/dataset/convert2torch.py

https://github.com/musyoku/gqn-datasets-translator/blob/master/convert.py
"""

import argparse
import collections
import functools
import gzip
import multiprocessing as mp
import os
import pathlib

import tensorflow as tf
import torch


DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)


_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)
_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5


def convert_record(path: pathlib.Path, dataset_name: str,
                   save_dir: pathlib.Path, batch_size: int, first_n: int
                   ) -> None:
    """Main process for one tfrecord file.

    This method load one tfrecord file, and preprocess each (frames, cameras)
    pair. The maximum number of pairs are bounded by `batch_size`.

    Args:
        path (pathlib.Path): Path to original data.
        dataset_name (str): Name of dataset.
        save_dir (pathlib.Path): Path to saved data.
        batch_size (int): Batch size of dataset for each tfrecord.
        first_n (int): Number of data to read (-1 means all).
    """

    # Dataset info
    dataset_info = _DATASETS[dataset_name]

    # Load tfrecord
    dataset = tf.data.TFRecordDataset(str(path))

    # Preprocess for each data and save to gzip file
    scene_list = []
    batch = 1
    for i, raw_data in enumerate(dataset.take(first_n)):
        scene_list.append(_preprocess_data(dataset_info, raw_data))

        # Save batch to a single file
        if (i + 1) % batch_size == 0:
            save_path = save_dir / f"{path.stem}-{batch}.pt.gz"
            with gzip.open(str(save_path), "wb") as f:
                torch.save(scene_list, f)

            scene_list = []
            batch += 1
    else:
        # Save rest
        if scene_list:
            save_path = save_dir / f"{path.stem}-{batch}.pt.gz"
            with gzip.open(str(save_path), "wb") as f:
                torch.save(scene_list, f)


def _preprocess_data(dataset_info: DatasetInfo, raw_data: tf.Tensor):
    """Converts raw data to tensor and saves into torch gziped file.

    Args:
        dataset_info (DatasetInfo): Information tuple.
        raw_data (tf.Tensor): Tensor of original data.
    """

    feature_map = {
        'frames': tf.io.FixedLenFeature(
            shape=dataset_info.sequence_size, dtype=tf.string),
        'cameras': tf.io.FixedLenFeature(
            shape=dataset_info.sequence_size * _NUM_RAW_CAMERA_PARAMS,
            dtype=tf.float32),
    }
    example = tf.io.parse_single_example(raw_data, feature_map)
    frames = _preprocess_frames(dataset_info, example)
    cameras = _preprocess_cameras(dataset_info, example)

    return frames.numpy().squeeze(), cameras.numpy().squeeze()


def _convert_frame_data(jpeg_data):
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


def _preprocess_frames(dataset_info, example):
    frames = tf.concat(example["frames"], axis=0)
    frames = tf.nest.map_structure(
        tf.stop_gradient,
        tf.map_fn(_convert_frame_data, tf.reshape(frames, [-1]),
                  dtype=tf.float32)
    )
    dataset_image_dimensions = tuple(
        [dataset_info.frame_size] * 2 + [_NUM_CHANNELS])
    frames = tf.reshape(
        frames, (-1, dataset_info.sequence_size) + dataset_image_dimensions)

    # Squeeze images to 64x64
    if dataset_info.frame_size != 64:
        frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
        new_frame_dimensions = (64,) * 2 + (_NUM_CHANNELS,)
        frames = tf.image.resize(frames, new_frame_dimensions[:2])
        frames = tf.reshape(
            frames, (-1, dataset_info.sequence_size) + new_frame_dimensions)

    return frames


def _preprocess_cameras(dataset_info, example):
    raw_pose_params = example["cameras"]
    cameras = tf.reshape(
        raw_pose_params,
        [-1, dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
    return cameras


def main():
    # Specify dataset name
    parser = argparse.ArgumentParser(description="Convert tfrecord to torch")
    parser.add_argument("--dataset", type=str,
                        default="shepard_metzler_5_parts",
                        help="Dataset name.")
    parser.add_argument("--mode", type=str, default="train",
                        help="Mode {train, test}.")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size of tfrecords.")
    parser.add_argument("--first-n", type=int, default=10,
                        help="Read only first n data in single a record "
                             "(-1 means all data).")
    args = parser.parse_args()

    if args.dataset not in _DATASETS:
        raise ValueError(f"Unrecognized dataset name {args.dataset}. ",
                         f"Available datasets are {_DATASETS.keys()}.")

    # Path
    root = pathlib.Path(os.getenv("DATA_DIR", "./data/"))
    tf_dir = root / f"{args.dataset}/{args.mode}/"
    torch_dir = root / f"{args.dataset}_torch/{args.mode}/"
    torch_dir.mkdir(parents=True, exist_ok=True)

    if not tf_dir.exists():
        raise FileNotFoundError(f"TFRecord path `{tf_dir}` does not exists.")

    # File list of original dataset
    record_list = sorted(tf_dir.glob("*.tfrecord"))

    # Multi process
    num_proc = mp.cpu_count()
    with mp.Pool(processes=num_proc) as pool:
        f = functools.partial(convert_record, dataset_name=args.dataset,
                              save_dir=torch_dir, batch_size=args.batch_size,
                              first_n=args.first_n)
        pool.map(f, record_list)


if __name__ == "__main__":
    main()
