'''
@Author : Yan Shipeng, Xie Jiangwei
@Contact: yanshp@shanghaitech.edu.cn, xiejw@shanghaitech.edu.cn
'''

import sys
import os
import os.path as osp
import copy
import time
import shutil
import cProfile
import logging
from pathlib import Path
import numpy as np
import random
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

repo_name = 'DER-ClassIL.pytorch'
base_dir = osp.realpath(".")[:osp.realpath(".").index(repo_name) + len(repo_name)]
sys.path.insert(0, base_dir)

from sacred import Experiment
ex = Experiment(base_dir=base_dir)

# Save which files
# ex.add_source_file(osp.join(base_dir, "inclearn/models/icarl.py"))
# ex.add_source_file(osp.join(base_dir, "inclearn/lib/data.py"))
# ex.add_source_file(osp.join(base_dir, "inclearn/lib/network.py"))
# ex.add_source_file(osp.join(base_dir, "inclearn/convnet/resnet.py"))
# ex.add_source_file(osp.join(os.getcwd(), "icarl.py"))
# ex.add_source_file(osp.join(os.getcwd(), "network.py"))
# ex.add_source_file(osp.join(os.getcwd(), "resnet.py"))

# MongoDB Observer
# ex.observers.append(MongoObserver.create(url='xx.xx.xx.xx:port', db_name='classil'))

import torch

from inclearn.tools import factory, results_utils, utils
from inclearn.learn.pretrain import pretrain
from inclearn.tools.metrics import IncConfusionMeter

def initialization(config, seed, mode, exp_id):
    # Add it if your input size is fixed
    # ref: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.benchmark = True  # This will result in non-deterministic results.
    # ex.captured_out_filter = lambda text: 'Output capturing turned off.'
    cfg = edict(config)
    utils.set_seed(cfg['seed'])
    if exp_id is None:
        exp_id = -1
        cfg.exp.savedir = "./logs"
    logger = utils.make_logger(f"exp{exp_id}_{cfg.exp.name}_{mode}", savedir=cfg.exp.savedir)

    # Tensorboard
    exp_name = f'{exp_id}_{cfg["exp"]["name"]}' if exp_id is not None else f'../inbox/{cfg["exp"]["name"]}'
    tensorboard_dir = cfg["exp"]["tensorboard_dir"] + f"/{exp_name}"

    # If not only save latest tensorboard log.
    # if Path(tensorboard_dir).exists():
    #     shutil.move(tensorboard_dir, cfg["exp"]["tensorboard_dir"] + f"/../inbox/{time.time()}_{exp_name}")

    tensorboard = SummaryWriter(tensorboard_dir)

    return cfg, logger, tensorboard


@ex.command
def train(_run, _rnd, _seed):
    cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "train", _run._id)
    ex.logger.info(cfg)
    cfg.data_folder = osp.join(base_dir, "data")

    start_time = time.time()
    _train(cfg, _run, ex, tensorboard)
    ex.logger.info("Training finished in {}s.".format(int(time.time() - start_time)))


def _train(cfg, _run, ex, tensorboard):
    device = factory.set_device(cfg)
    trial_i = cfg['trial']

    inc_dataset = factory.get_data(cfg, trial_i)
    ex.logger.info("classes_order")
    ex.logger.info(inc_dataset.class_order)

    model = factory.get_model(cfg, trial_i, _run, ex, tensorboard, inc_dataset)

    if _run.meta_info["options"]["--file_storage"] is not None:
        _save_dir = osp.join(_run.meta_info["options"]["--file_storage"], str(_run._id))
    else:
        _save_dir = cfg["exp"]["ckptdir"]

    results = results_utils.get_template_results(cfg)

    for task_i in range(inc_dataset.n_tasks):
        task_info, train_loader, val_loader, test_loader = inc_dataset.new_task()

        model.set_task_info(
            task=task_info["task"],
            total_n_classes=task_info["max_class"],
            increment=task_info["increment"],
            n_train_data=task_info["n_train_data"],
            n_test_data=task_info["n_test_data"],
            n_tasks=inc_dataset.n_tasks,
        )

        model.before_task(task_i, inc_dataset)
        # TODO: Move to incmodel.py
        if 'min_class' in task_info:
            ex.logger.info("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))

        # Pretraining at step0 if needed
        if task_i == 0 and cfg["start_class"] > 0:
            do_pretrain(cfg, ex, model, device, train_loader, test_loader)
            inc_dataset.shared_data_inc = train_loader.dataset.share_memory
        elif task_i < cfg['start_task']:
            state_dict = torch.load(f'./ckpts/step{task_i}.ckpt')
            model._parallel_network.load_state_dict(state_dict)
            inc_dataset.shared_data_inc = train_loader.dataset.share_memory
        else:
            model.train_task(train_loader, val_loader)
        model.after_task(task_i, inc_dataset)

        ex.logger.info("Eval on {}->{}.".format(0, task_info["max_class"]))
        ypred, ytrue = model.eval_task(test_loader)
        acc_stats = utils.compute_accuracy(ypred, ytrue, increments=model._increments, n_classes=model._n_classes)

        #Logging
        model._tensorboard.add_scalar(f"taskaccu/trial{trial_i}", acc_stats["top1"]["total"], task_i)

        _run.log_scalar(f"trial{trial_i}_taskaccu", acc_stats["top1"]["total"], task_i)
        _run.log_scalar(f"trial{trial_i}_task_top5_accu", acc_stats["top5"]["total"], task_i)

        ex.logger.info(f"top1:{acc_stats['top1']}")
        ex.logger.info(f"top5:{acc_stats['top5']}")

        results["results"].append(acc_stats)

    top1_avg_acc, top5_avg_acc = results_utils.compute_avg_inc_acc(results["results"])

    _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top1"] = top1_avg_acc
    _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top5"] = top5_avg_acc
    ex.logger.info("Average Incremental Accuracy Top 1: {} Top 5: {}.".format(
        _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top1"],
        _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top5"],
    ))
    if cfg["exp"]["name"]:
        results_utils.save_results(results, cfg["exp"]["name"])


def do_pretrain(cfg, ex, model, device, train_loader, test_loader):
    if not os.path.exists(osp.join(ex.base_dir, 'pretrain/')):
        os.makedirs(osp.join(ex.base_dir, 'pretrain/'))
    model_path = osp.join(
        ex.base_dir,
        "pretrain/{}_{}_cosine_{}_multi_{}_aux{}_nplus1_{}_{}_trial_{}_{}_seed_{}_start_{}_epoch_{}.pth".format(
            cfg["model"],
            cfg["convnet"],
            cfg["weight_normalization"],
            cfg["der"],
            cfg["use_aux_cls"],
            cfg["aux_n+1"],
            cfg["dataset"],
            cfg["trial"],
            cfg["train_head"],
            cfg['seed'],
            cfg["start_class"],
            cfg["pretrain"]["epochs"],
        ),
    )
    if osp.exists(model_path):
        print("Load pretrain model")
        if hasattr(model._network, "module"):
            model._network.module.load_state_dict(torch.load(model_path))
        else:
            model._network.load_state_dict(torch.load(model_path))
    else:
        pretrain(cfg, ex, model, device, train_loader, test_loader, model_path)

@ex.command
def test(_run, _rnd, _seed):
    cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "test", _run._id)
    ex.logger.info(cfg)

    trial_i = cfg['trial']
    cfg.data_folder = osp.join(base_dir, "data")
    inc_dataset = factory.get_data(cfg, trial_i)
    # inc_dataset._current_task = taski
    # train_loader = inc_dataset._get_loader(inc_dataset.data_cur, inc_dataset.targets_cur)
    model = factory.get_model(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    model._network.task_size = cfg.increment

    test_results = results_utils.get_template_results(cfg)
    for taski in range(inc_dataset.n_tasks):
        task_info, train_loader, _, test_loader = inc_dataset.new_task()
        model.set_task_info(
            task=task_info["task"],
            total_n_classes=task_info["max_class"],
            increment=task_info["increment"],
            n_train_data=task_info["n_train_data"],
            n_test_data=task_info["n_test_data"],
            n_tasks=task_info["max_task"]
        )
        model.before_task(taski, inc_dataset)
        state_dict = torch.load(f'./ckpts/step{taski}.ckpt')
        model._parallel_network.load_state_dict(state_dict)
        model.eval()

        #Build exemplars
        model.after_task(taski, inc_dataset)

        ypred, ytrue = model.eval_task(test_loader)

        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=model._increments, n_classes=model._n_classes)
        test_results['results'].append(test_acc_stats)
        ex.logger.info(f"task{taski} test top1acc:{test_acc_stats['top1']}")

    avg_test_acc = results_utils.compute_avg_inc_acc(test_results['results'])
    ex.logger.info(f"Test Average Incremental Accuracy: {avg_test_acc}")

if __name__ == "__main__":
    # ex.add_config('./codes/base/configs/default.yaml')
    ex.add_config("./configs/default.yaml")
    ex.run_commandline()
