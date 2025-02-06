"""
Code for Teacher Student Mutual Adversarial Training
"""

import os
import logging
import argparse
import sys

import torch
import random
import numpy as np

from training.training_helpers import train_teacher_student_adv_ml, log_best_checkpoint_dict
from torch.utils.tensorboard import SummaryWriter
from data.datasets import T1SNPDataset
from utils.log_utils import set_logging_settings, add_common_args

parser = argparse.ArgumentParser()

# ######## Model Parameters
parser.add_argument('--gan_mode', default='vanilla', help='vanilla| lsgan | wgangp')
parser.add_argument('--T', default=2.)
parser.add_argument('--l1', default=1.)
parser.add_argument('--l2', default=1.)
parser.add_argument('--l3', default=1.)
parser.add_argument('--distill_confidence', default=None, help='if None, all the samples will be used for distill.')

# Student
parser.add_argument('--student_resnet', default='resnet-18')
parser.add_argument('--student_G_dropout', default=0.5)
parser.add_argument('--student_G_type', default=2)
parser.add_argument('--student_G_act', default='relu')
parser.add_argument('--student_G_lr', default=1e-4)
parser.add_argument('--student_G_betas', default=(0.9, 0.99))
parser.add_argument('--student_c_load_from_teacher', default=False)
parser.add_argument('--student_G_weight_decay', default=0.)
parser.add_argument('--student_detach_gene', default=False)
parser.add_argument('--student_class_lr_scheduler_patience', default=10)
parser.add_argument('--student_g_lr_scheduler_patience', default=10)
parser.add_argument('--discriminator_lr_scheduler_patience', default=10)
parser.add_argument('--student_class_lr_scheduler_factor', default=0.7)
parser.add_argument('--student_g_lr_scheduler_factor', default=0.7)
parser.add_argument('--discriminator_lr_scheduler_factor', default=0.7)

parser.add_argument('--embed_dim_gene', default=256)
parser.add_argument('--init_type', type=str, default='normal',
                    help='network initialization [default | normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--student_c_lr', default=1e-4)
parser.add_argument('--student_c_betas', default=(0.9, 0.99))
parser.add_argument('--student_c_weight_decay', default=1e-4)
parser.add_argument('--num_class', default=3)
# Teacher
parser.add_argument('--teacher_resnet', default='resnet-18')
parser.add_argument('--teacher_gene_net_dropout', default=0.5)
parser.add_argument('--teacher_alpha', default=0.999)
parser.add_argument('--teacher_optimizer', default='adamw', help='adamw, sgd')
parser.add_argument('--lr', default=0.005)
parser.add_argument('--teacher_weight_decay', default=1)
parser.add_argument('--teacher_lasso', default=0.)
parser.add_argument('--teacher_dropout', default=0.)
parser.add_argument('--early_stopping_patience', default=50)
parser.add_argument('--teacher_initialization', type=str, default='default')

# Discriminator
parser.add_argument('--netD_lr', default=1e-4)
parser.add_argument('--netD_betas', default=(0.9, 0.99))
parser.add_argument('--netD_weight_decay', default=0.)
parser.add_argument('--pretr_teacher_dir', required=True)
parser = add_common_args(parser)
args = parser.parse_args()

args.data_split_ratio = [0.8, 0.0, 0.2]

# ############################# Logging & Fixing Seed #############################
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if int(args.gpus) >= 0:
    torch.cuda.manual_seed_all(args.random_seed)

args = set_logging_settings(args, os.path.basename(__file__).split('.')[0])
args.writer = SummaryWriter(args.stdout)

############################# Loading Data #############################
if args.debug:
    args.batch_size_train = 2
    args.num_epochs = 20

dataset = T1SNPDataset(t1_data_path=args.t1_file, cognitive_data_path=args.cognitive_scores_file,
                       snp_data_path=args.snp_file, target=args.target)

args.snp_num_features = dataset.snp_data_num_features
args.t1_num_features = dataset.t1_data_num_features

print(args.pretr_teacher_dir)
args.target = args.pretr_teacher_dir.split('/')[-2][:7]
args.teacher_snp_net_backbone, args.teacher_t1_net_backbone, args.teacher_t1_n_blocks, \
    args.teacher_snp_n_blocks = args.pretr_teacher_dir.split('/')[-2].split('_')[2:6]

args.teacher_t1_n_blocks = int(args.teacher_t1_n_blocks)
args.teacher_snp_n_blocks = int(args.teacher_snp_n_blocks)

print(args.target, args.teacher_snp_net_backbone, args.teacher_t1_net_backbone, args.teacher_t1_n_blocks,
      args.teacher_snp_n_blocks)

logging.warning('loading checkpoint from: {}'.format(args.pretr_teacher_dir))

checkpoint_class, best_scores, best_accs, best_prs, best_rcs, best_f1s = train_teacher_student_adv_ml(dataset, args)

log_best_checkpoint_dict(checkpoint_class)

checkpoint_class_name = 'class_checkpoint_iter_{}.pth'.format(checkpoint_class['iter'])
file_name_class = os.path.join(args.checkpoint_dir, checkpoint_class_name)
logging.warning('Saving Checkpoint: {}'.format(file_name_class))
torch.save(checkpoint_class, file_name_class)

if args.task == 'classification':
    logging.warning('Best Scores:\nacc_mean:{} acc_std:{}\npr_mean:{}, pr_std:{}\n'
                    'rc_mean:{},rc_std:{}\nf1_mean:{},f1_std:{}'.format(torch.mean(torch.tensor(best_accs)),
                                                                        torch.std(torch.tensor(best_accs)),
                                                                        torch.mean(torch.tensor(best_prs)),
                                                                        torch.std(torch.tensor(best_prs)),
                                                                        torch.mean(torch.tensor(best_rcs)),
                                                                        torch.std(torch.tensor(best_rcs)),
                                                                        torch.mean(torch.tensor(best_f1s)),
                                                                        torch.std(torch.tensor(best_f1s))
                                                                        ))
else:
    logging.warning('Best Scores: mean:{} std:{}'.format(torch.mean(torch.tensor(best_scores)),
                                                         torch.std(torch.tensor(best_scores))))
