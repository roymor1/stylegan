# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Main entry point for training StyleGAN and ProGAN networks."""
import argparse
import copy
import dnnlib
from dnnlib import EasyDict

import config
from metrics import metric_base

########################################################################################################################
# Constants
########################################################################################################################
MINI_BATCH_DICT = {
    1: {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4},
    2: {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8},
    4: {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16},
    8: {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}
}

########################################################################################################################
# App
########################################################################################################################
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', default = '/media/hd1/projects/stylegan/ffhq-256/exp001')
    parser.add_argument('-d', '--datadir', default = '/media/hd1/data/ffhq/tfrecords')
    parser.add_argument('--tfrecord_dirname', default = 'ffhq')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--total_kimg', type=int, default=25000)
    parser.add_argument('--resume_run_id',
                        default=None, #'latest',
                        help='Run ID or network pkl to resume training from, None = start from scratch.')
    parser.add_argument('--resume_snapshot',
                        default=None,
                        help='Snapshot index to resume training from, None = autodetect.')
    parser.add_argument('--resume_kimg', type=float,
                        default= 0.0,
                        help='Assumed training progress at the beginning. Affects reporting and training schedule.')
    parser.add_argument('--resume_time', type=float,
                        default= 0.0,
                        help='Assumed wallclock time at the beginning. Affects reporting.')
    return parser.parse_args()


def main(args):
    desc = 'sgan'  # Description string included in result subdir name.
    train = EasyDict(run_func_name='training.training_loop.training_loop')  # Options for training loop.
    G = EasyDict(func_name='training.networks_stylegan.G_style')  # Options for generator network.
    D = EasyDict(func_name='training.networks_stylegan.D_basic')  # Options for discriminator network.
    G_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)  # Options for generator optimizer.
    D_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)  # Options for discriminator optimizer.
    G_loss = EasyDict(func_name='training.loss.G_logistic_nonsaturating')  # Options for generator loss.
    D_loss = EasyDict(func_name='training.loss.D_logistic_simplegp', r1_gamma=10.0)  # Options for discriminator loss.
    dataset = EasyDict(tfrecord_dir=args.tfrecord_dirname, resolution=args.resolution)
    desc += '-ffhq'+ str(args.resolution)
    sched = EasyDict()  # Options for TrainingSchedule.
    grid = EasyDict(size='4k', layout='random')  # Options for setup_snapshot_image_grid().
    metrics = [metric_base.fid50k]  # Options for MetricGroup.
    submit_config = dnnlib.SubmitConfig()  # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}  # Options for tflib.init_tf().
    train.mirror_augment = True

    # Number of GPUs.
    submit_config.num_gpus = args.n_gpus
    desc += '-{}gpu'.format(args.n_gpus)
    sched.minibatch_base = 4 * args.n_gpus
    sched.minibatch_dict = MINI_BATCH_DICT[args.n_gpus]

    # Default options.
    train.total_kimg = args.total_kimg
    sched.lod_initial_resolution = 8
    sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)

    # kwargs
    kwargs = EasyDict(train)
    kwargs.update(data_root_dir=args.datadir)
    kwargs.update(resume_run_id=args.resume_run_id,
                  resume_snapshot=args.resume_snapshot,
                  resume_kimg=args.resume_kimg,
                  resume_time=args.resume_time)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.run_dir_root = args.outdir
    kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)


if __name__ == "__main__":
    main(get_args())
