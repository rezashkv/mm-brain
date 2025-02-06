"""
Code for pretraining the teacher model.
"""

import os
import logging
import argparse
import torch
import random
import numpy as np
from training.training_helpers import pretrain_teacher_model
from torch.utils.tensorboard import SummaryWriter
from data.datasets import T1SNPDataset
from utils.log_utils import set_logging_settings, add_common_args


parser = argparse.ArgumentParser()

# Added August 24 - Start
parser.add_argument('--functional_regularization', default=False)
parser.add_argument('--perturbation_strength', default=0.1)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--norm', type=int, default=1)
parser.add_argument('--estimation', type=str, default='ent')
parser.add_argument('--optim_method', type=str, default='max_ent_minus', help='max_ent, max_ent_minus')
parser.add_argument('--n_samples', type=int, default=3)
parser.add_argument('--grad', default=True)
# Added August 24 - End

# Teacher Parameters
parser.add_argument('--multimodal_teacher', default=False)

parser.add_argument('--teacher_lasso', default=0.)
parser.add_argument('--teacher_dropout', default=0.1)
parser.add_argument('--early_stopping_patience', default=50)
parser.add_argument('--teacher_initialization', type=str, default='default')

parser = add_common_args(parser)
args = parser.parse_args()

# ############################# Logging & Fixing Seed #############################
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if int(args.gpus) >= 0:
    torch.cuda.manual_seed_all(args.random_seed)

args = set_logging_settings(args, os.path.basename(__file__).split('.')[0])
args.writer = SummaryWriter(args.stdout)


# ############################# Loading Data #############################
if args.debug:
    args.batch_size_train = 8
    args.num_epochs = 1

dataset = T1SNPDataset(t1_data_path=args.t1_file, cognitive_data_path=args.cognitive_scores_file,
                       snp_data_path=args.snp_file, target=args.target)


args.snp_num_features = dataset.snp_data_num_features
args.t1_num_features = dataset.t1_data_num_features


_, best_scores, _, _, _, _ = pretrain_teacher_model(dataset, args)

logging.warning('Best Scores: mean:{} std:{}'.format(torch.mean(torch.tensor(best_scores)),
                                                             torch.std(torch.tensor(best_scores))))
for i, score in enumerate(best_scores):
    logging.warning('Fold: {}, loss: {}'.format(i, score))


