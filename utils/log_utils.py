import os
import sys
import logging
import time
import socket
import torch

from argparse import ArgumentParser
from .allenNLP_tee_logger import TeeLogger


def prepare_global_logging(serialization_dir: str, file_friendly_logging: bool) -> logging.FileHandler:
    """
    This function configures 3 global logging attributes - streaming stdout and stderr
    to a file as well as the terminal, setting the formatting for the python logging
    library and setting the interval frequency for the Tqdm progress bar.

    Note that this function does not set the logging level, which is set in ``allennlp/run.py``.

    Parameters
    ----------
    serialization_dir : ``str``, required.
        The directory to stream logs to.
    file_friendly_logging : ``bool``, required.
        Whether logs should clean the output to prevent carriage returns
        (used to update progress bars on a single terminal line). This
        option is typically only used if you are running in an environment
        without a terminal.

    Returns
    -------
    ``logging.FileHandler``
        A logging file handler that can later be closed and removed from the global logger.
    """

    # If we don't have a terminal as stdout,
    # force tqdm to be nicer.
    if not sys.stdout.isatty():
        file_friendly_logging = True

    # Tqdm.set_slower_interval(file_friendly_logging)
    std_out_file = os.path.join(serialization_dir, "stdout.log")
    sys.stdout = TeeLogger(std_out_file,  # type: ignore
                           sys.stdout,
                           file_friendly_logging)
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"),  # type: ignore
                           sys.stderr,
                           file_friendly_logging)

    stdout_handler = logging.FileHandler(std_out_file)
    stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(stdout_handler)

    return stdout_handler


def logging_args(args):
    args_items = vars(args)
    args_keys = list(args_items.keys())
    args_keys.sort()
    for arg in args_keys:
        logging.warning(str(arg) + ': ' + str(getattr(args, arg)))


def set_logging_settings(args, main_file):
    args.data_dir = 'path/to/data'
    args.logs_dir = 'path/to/logs'
    args.stdout = 'path/to/stdout'

    args.snp_file = os.path.join(args.data_dir, args.snp_file)
    args.t1_file = os.path.join(args.data_dir, args.t1_file)
    args.cognitive_scores_file = os.path.join(args.data_dir, args.cognitive_scores_file)

    args.device = torch.device(f"cuda:{args.device_ids}" if (torch.cuda.is_available()) else "cpu")
    if 'cuda' in args.device.type:
        args.num_workers = 16
    else:
        args.num_workers = 0
    device_type = args.device.type
    index = str(args.device.index) if (isinstance(args.device.index, int)) else ''
    args.stdout = os.path.join(args.stdout, os.path.basename(os.getcwd()), main_file,
                               time.strftime("%d-%m-%Y_") + time.strftime("%H-%M-%S_") + socket.gethostname() + '_' +
                               device_type + index)
    checkpoint_folder = os.path.join(args.logs_dir, 'checkpoints')
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    checkpoint_dir = os.path.join(checkpoint_folder, main_file)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    args.checkpoint_dir = os.path.join(checkpoint_dir, '_'.join([args.target, args.teacher_snp_net_backbone,
                                                                 args.teacher_t1_net_backbone,
                                                                 str(args.teacher_t1_n_blocks),
                                                                 str(args.teacher_snp_n_blocks)]))

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=args.logging_level)
    prepare_global_logging(serialization_dir=args.stdout, file_friendly_logging=False)
    if args.verbose > 0:
        logging_args(args)
    return args


def add_common_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # General Experimental Settings
    parser.add_argument('--task', type=str, default='regression')
    parser.add_argument('--random_seed', type=int, default=43)
    parser.add_argument('--verbose', default=1)
    parser.add_argument('--user', default='user')
    parser.add_argument('--gpus', default='1', help='gpu:i, i in [0, 1, 2, 3]')
    parser.add_argument('--device_ids', default='0')
    parser.add_argument('--debug', default=False, help='loads less training data to make debug faster.')
    parser.add_argument('--debug_len', default=16, help='loads 2 samples in train/val/test datasets.')
    parser.add_argument('--num_epochs', default=400)
    parser.add_argument('--saving_freq', default=1)
    parser.add_argument('--num_test_epochs', default=1)
    parser.add_argument('--batch_size_train', default=64)
    parser.add_argument('--k_fold', default=5)
    parser.add_argument('--target_dim', default=1)

    # Loss & Metric Parameters
    parser.add_argument('--gamma_focal', default=2., help='gamma for focal loss.')
    parser.add_argument('--metric', default='loss', help='auc, loss')

    # Files Information.
    parser.add_argument('--logging_level', default=logging.INFO, help='Options: logging.DEBUG, logging.INFO')
    parser.add_argument('--snp_file', type=str, default='new_snp_data.csv')
    parser.add_argument('--t1_file', type=str, default='new_t1_data.csv')
    parser.add_argument('--cognitive_scores_file', type=str, default='new_cognitive_data-first.csv')
    parser.add_argument('--target', type=str, default='PHC_MEM')

    # Data Parameters
    parser.add_argument('--standardize_data', default=False)
    parser.add_argument('--num_workers', default=16)
    parser.add_argument('--seed_data', default=1, help='random seed used for partitioning the data.')
    parser.add_argument('--data_split_ratio', default=[0.9, 0.0, 0.1])

    # Teacher Parameters
    parser.add_argument('--teacher_snp_net_backbone', type=str, default='mlp', help='options: mlp, transformer, resnet')
    parser.add_argument('--teacher_t1_net_backbone', type=str, default='mlp',
                        help='options: mlp, transformer, resnet')
    parser.add_argument('--teacher_t1_n_blocks', type=int, default=2)
    parser.add_argument('--teacher_snp_n_blocks', type=int, default=2)

    parser.add_argument('--teacher_snp_net_d_layers', default=[512, 512, 512])
    parser.add_argument('--teacher_snp_net_dropout', default=0.1)
    parser.add_argument('--teacher_snp_net_d_out', default=256)
    parser.add_argument('--teacher_snp_net_batch_norm', default=True)
    parser.add_argument('--teacher_snp_net_activation', default='ReLU')
    parser.add_argument('--teacher_t1_net_d_layers', default=[256, 256, 256])
    parser.add_argument('--teacher_t1_net_dropout', default=0.1)
    parser.add_argument('--teacher_t1_net_d_out', default=256)
    parser.add_argument('--teacher_t1_net_batch_norm', default=True)
    parser.add_argument('--teacher_t1_net_activation', default='ReLU')

    # Optimizer Parameters
    parser.add_argument('--betas', default=(0.9, 0.99))
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--teacher_t1_optimizer', default='adamw', help='adamw, sgd')
    parser.add_argument('--teacher_snp_optimizer', default='adamw', help='adamw, sgd')
    parser.add_argument('--t1_lr', default=0.05)
    parser.add_argument('--snp_lr', default=0.05)
    parser.add_argument('--teacher_t1_weight_decay', default=0.1)
    parser.add_argument('--teacher_snp_weight_decay', default=0.1)
    parser.add_argument('--lr_scheduler_verbose', default=10)
    parser.add_argument('--t1_lr_scheduler_patience', default=10)
    parser.add_argument('--snp_lr_scheduler_patience', default=10)
    parser.add_argument('--t1_lr_scheduler_factor', default=0.7)
    parser.add_argument('--snp_lr_scheduler_factor', default=0.7)

    return parser
