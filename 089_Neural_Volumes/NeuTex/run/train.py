# import open3d
import sys
import os
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), ".."))

import copy
import torch
import numpy as np
import time
from options import TrainOptions
from data import create_data_loader, create_dataset
from models import create_model
from utils.visualizer import Visualizer
from utils import format as fmt


def test(model, dataset, visualizer, opt, test_steps):

    print(
        "-----------------------------------Testing-----------------------------------"
    )
    model.eval()
    total_num = len(dataset)
    select_ids = np.random.choice(range(total_num), opt.test_num)
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size

    height = dataset.height
    width = dataset.width

    for i in range(opt.test_num):
        data = dataset.get_item(select_ids[i])
        raydir = data["raydir"].clone()
        gt_image = data["gt_image"].clone()
        if "gt_depth" in data:
            gt_depth = data["gt_depth"].clone()
            gt_mask = data["gt_mask"].clone()

        if "gt_normal" in data:
            gt_normal = data["gt_normal"].clone()

        campos = data["campos"]
        raydir = data["raydir"]

        visuals = None
        for k in range(0, height * width, chunk_size):
            print(
                "chunk {}/{}".format(k // chunk_size + 1, height * width // chunk_size)
            )
            start = k
            end = min([k + chunk_size, height * width])

            data["raydir"] = raydir[:, start:end, :]
            data["gt_image"] = gt_image[:, start:end, :]
            if "gt_depth" in data:
                data["gt_depth"] = gt_depth[:, start:end]
                data["gt_mask"] = gt_mask[:, start:end]
            if "gt_normal" in data:
                data["gt_normal"] = gt_normal[:, start:end]
            model.set_input(data)
            model.test()
            curr_visuals = model.get_current_visuals()

            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    chunk = value.cpu().numpy()
                    if len(chunk.shape) == 2:
                        visuals[key] = np.zeros((height * width)).astype(chunk.dtype)
                        visuals[key][start:end] = chunk
                    else:
                        assert len(chunk.shape) == 3
                        assert chunk.shape[-1] == 3
                        visuals[key] = np.zeros((height * width, 3)).astype(chunk.dtype)
                        visuals[key][start:end, :] = chunk
            else:
                for key, value in curr_visuals.items():
                    visuals[key][start:end] = value.cpu().numpy()

        for key, value in visuals.items():
            visuals[key] = visuals[key].reshape(height, width, -1).squeeze()
        visualizer.display_current_results(
            visuals, test_steps, campos.squeeze(), raydir.squeeze()
        )
        test_steps = test_steps + 1

    model.train()
    print(
        "--------------------------------Finish Testing--------------------------------"
    )
    return


def main():
    torch.backends.cudnn.benchmark = True
    opt = TrainOptions().parse()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        print(
            fmt.RED + "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )
        print("Debug Mode")
        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" + fmt.END
        )

    data_loader = create_data_loader(opt)
    dataset_size = len(data_loader)
    print("# training images = {}".format(dataset_size))

    if opt.resume_dir:
        resume_dir = opt.resume_dir
        states = torch.load(
            os.path.join(resume_dir, "{}_states.pth".format(opt.resume_epoch))
        )
        epoch_count = states["epoch_count"]
        total_steps = states["total_steps"]
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Continue training from {} epoch".format(opt.resume_epoch))
        print("Iter: ", total_steps)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    else:
        epoch_count = 1
        total_steps = 0

    # load model
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    # create test loader
    test_opt = copy.deepcopy(opt)
    test_opt.is_train = False
    test_opt.random_sample = "no_crop"
    test_opt.batch_size = 1
    test_opt.n_threads = 0
    test_dataset = create_dataset(test_opt)

    with open("/tmp/.neural-volumetric.name", "w") as f:
        f.write(opt.name + "\n")

    visualizer.reset()
    # total_epochs = (opt.niter + opt.niter_decay) * opt.batch_size // dataset_size + 1
    # for epoch in range(epoch_count, opt.niter + opt.niter_decay + 1):
    # for epoch in range(epoch_count, total_epochs + 1):
    epoch = epoch_count
    stop_iteration = False
    while not stop_iteration:
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(data_loader):
            total_steps += 1
            epoch_iter += 1

            if total_steps > opt.niter + opt.niter_decay:
                stop_iteration = True
                break

            model.set_input(data)
            model.set_current_step(total_steps)
            model.optimize_parameters()
            model.update_learning_rate(verbose=total_steps % opt.print_freq == 0)

            losses = model.get_current_losses()
            visualizer.accumulate_losses(losses)

            if total_steps and total_steps % opt.print_freq == 0:
                print(
                    "Total iterations {}/{}".format(
                        total_steps, opt.niter + opt.niter_decay
                    )
                )
                visualizer.print_losses(total_steps)
                visualizer.reset()

            if opt.train_and_test == 1 and total_steps % opt.test_freq == 0:
                test(model, test_dataset, visualizer, test_opt, total_steps)

            try:
                if total_steps % opt.save_iter_freq == 0:
                    other_states = {"epoch_count": epoch, "total_steps": total_steps}
                    print(
                        "saving model (epoch {}, total_steps {})".format(
                            epoch, total_steps
                        )
                    )
                    model.save_networks(total_steps, other_states)
                    model.save_networks("latest", other_states)
                    model.save_subnetworks("latest")
            except Exception as e:
                print(e)

        print(
            "{}: End of epoch {} \t Time Taken: {} sec".format(
                opt.name, epoch, time.time() - epoch_start_time,
            )
        )
        epoch += 1

    try:
        other_states = {"epoch_count": epoch, "total_steps": total_steps}
        model.save_networks("latest", other_states)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
