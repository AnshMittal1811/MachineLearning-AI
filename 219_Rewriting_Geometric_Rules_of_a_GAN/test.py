import argparse
import pandas as pd
from collections import OrderedDict

import models
import datasets
from models.reference_model import ReferenceModel
from evaluation import GroupEvaluator
from option.option import TestOptions


def test_batch(all_tasks, eval_metrics, out_dir):
    output_file = f'{out_dir}/table.csv'

    all_metrics = OrderedDict()
    table_columns = ['test_psnr', 'test_psnr_changed', 'test_psnr_unchanged', 'test_ssim', 'test_ssim_changed', 'test_ssim_unchanged', 'test_lpips', 'test_lpips_changed', 'test_lpips_unchanged', 'test_chamfer', 'test_chamfer_changed', 'test_chamfer_unchanged']

    for name, dset, pretrained, exp_name in all_tasks:
        pretrained_pkl = pretrained_models[pretrained]
        command = f'python test.py --dataroot ./data/{dset} --model_path {exp_name} --pretrained_G {pretrained_pkl} --evaluation_metrics {eval_metrics} --result_dir {out_dir}'
        if pretrained in ['horse', 'house']:
            command += ' --random_sample_trunc 0.5'

        opt = TestOptions().parse(command=command)
        model = models.create_model(opt)
        model.load(load_path=opt.model_path)
        ref_model = ReferenceModel(opt)

        # update option
        opt.image_res = model.get_output_res()
        opt.target_res = model.get_target_res()

        # some bookkeeping
        dataset = datasets.create_dataset(opt)
        evaluators = GroupEvaluator(opt)

        # evaluation and visualization (e.g., feature visualization, random samples)
        models_to_eval = (ref_model, model)
        metrics, visuals = evaluators.evaluate(models_to_eval, dataset)

        all_metrics[name] = metrics
        print(f"({name}) {all_metrics[name]}")
        table = pd.DataFrame.from_dict(all_metrics, orient='index', columns=table_columns)
        table.to_csv(output_file, na_rep='--')

# TODO: handle opt model_path
def parse_task_file(task_file):
    with open(task_file, 'r') as f:
        all_tasks = [s.split() for s in f.readlines()]
    return all_tasks


pretrained_models = {
    'cat': './pretrained/stylegan3-r-afhqv2cat-512x512.pkl',
    'horse': './pretrained/stylegan3-r-horse-256x256.pkl',
    'house': './pretrained/stylegan3-r-house-512x512.pkl',
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tasks', required=True)
    parser.add_argument('-d', '--out_dir', default='./results/metrics')
    parser.add_argument('-m', '--eval_metrics', default='pixelerror,chamfer,sample')
    args = parser.parse_args()

    all_tasks = parse_task_file(args.tasks)
    test_batch(all_tasks, args.eval_metrics, args.out_dir)
