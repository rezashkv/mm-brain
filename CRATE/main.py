import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import sys
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset
import math
from model.crate import *
from model.vit import *
from data.dataset import T1SNPDataset
from lion_pytorch import Lion
from sklearn.metrics import r2_score, f1_score, precision_score, recall_score
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from losses.focal_loss import StaticFocalLoss

model_names = ["vit_tiny", "vit_small", "CRATE_tiny", "CRATE_small", "CRATE_base", "CRATE_large",
               "CRATE_tabular_tiny", "CRATE_tabular_small", "CRATE_tabular_base", "CRATE_tabular_large",
               "Node", "MLP", "ResNet", "SNN", "single_transformer_tabular_crate_large",
               "inter_attention_single_transformer_tabular_crate_large"
               ]


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR', default="/path/to/imagenet",
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='CRATE_tiny',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: CRATE_tiny)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--label_smooth', default=0.1, type=float, metavar='L',
                        help='label smoothing coef')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.0004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--optimizer', default="AdamW", type=str,
                        help='Optimizer to Use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
    parser.add_argument('--task', default='classification', type=str, help='task to perform')
    parser.add_argument('--target', default='PHC_MEM', type=str, help='target cognitive score')
    parser.add_argument('--runs', nargs='+', type=int, default=[0], help='random seeds')

    return parser


parser = get_args_parser()
best_acc1 = 0

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()


def main():
    args = parser.parse_args()
    if args.task == "classification":
        args.num_classes = 3
    else:
        args.num_classes = 1

    val_best_accuracies, val_best_losses, val_best_f1s, val_best_precisions, val_best_recalls = [], [], [], [], []
    for run in args.runs:
        print(f"########## Run {run} ##########")
        args.seed = run

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            cudnn.benchmark = False
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        if args.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')

        if args.dist_url == "env://" and args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])

        args.distributed = args.world_size > 1 or args.multiprocessing_distributed

        if torch.cuda.is_available():
            ngpus_per_node = torch.cuda.device_count()
        else:
            ngpus_per_node = 1
        if args.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.world_size = ngpus_per_node * args.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            # Simply call main_worker function
            val_best_acc, val_best_loss, val_best_f1, val_best_precision, val_best_recall = (
                main_worker(args.gpu, ngpus_per_node, args))
            val_best_accuracies.append(val_best_acc)
            val_best_losses.append(val_best_loss)
            val_best_f1s.append(val_best_f1)
            val_best_precisions.append(val_best_precision)
            val_best_recalls.append(val_best_recall)

    # report mean and std of the validation scores
    warnings.warn(f"Model: {args.arch} on {args.target}")
    warnings.warn(f"Mean +- Std Val Acc: {sum(val_best_accuracies) / len(val_best_accuracies)} +- "
                  f"{torch.tensor(val_best_accuracies).std()}")
    warnings.warn(f"Mean +- Std Val Loss: {sum(val_best_losses) / len(val_best_losses)} +- "
                  f"{torch.tensor(val_best_losses).std()}")
    warnings.warn(f"Mean +- Std Val F1: {sum(val_best_f1s) / len(val_best_f1s)} +- "
                  f"{torch.tensor(val_best_f1s).std()}")
    warnings.warn(f"Mean +- Std Val Precision: {sum(val_best_precisions) / len(val_best_precisions)} +- "
                  f"{torch.tensor(val_best_precisions).std()}")
    warnings.warn(f"Mean +- Std Val Recall: {sum(val_best_recalls) / len(val_best_recalls)} +- "
                  f"{torch.tensor(val_best_recalls).std()}")


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    args.tabular = "tabular" in args.arch.lower() or "node" in args.arch.lower() or "mlp" in args.arch.lower() or \
                   "resnet" in args.arch.lower() or "snn" in args.arch.lower()

    print('==> Building model: {}'.format(args.arch))
    if args.arch == 'vit_tiny':
        model = vit_tiny_patch16(global_pool=True)
    elif args.arch == 'vit_small':
        model = vit_small_patch16(global_pool=True)
    elif args.arch == 'CRATE_tiny':
        model = CRATE_tiny(num_classes=args.num_classes)
    elif args.arch == "CRATE_small":
        model = CRATE_small(num_classes=args.num_classes)
    elif args.arch == "CRATE_base":
        model = CRATE_base(num_classes=args.num_classes)
    elif args.arch == "CRATE_large":
        model = CRATE_large(num_classes=args.num_classes)
    elif args.arch == "CRATE_tabular_tiny":
        model = CRATE_tabular_tiny(num_classes=args.num_classes)
    elif args.arch == "CRATE_tabular_small":
        model = CRATE_tabular_small(num_classes=args.num_classes)
    elif args.arch == "CRATE_tabular_base":
        model = CRATE_tabular_base(num_classes=args.num_classes)
    elif args.arch == "CRATE_tabular_large":
        model = CRATE_tabular_large(num_classes=args.num_classes)
    elif args.arch == "Node":
        model = NODE_baseline(num_classes=args.num_classes)
    elif args.arch == "MLP":
        model = MLP_baseline(num_classes=args.num_classes)
    elif args.arch == "ResNet":
        model = ResNet_baseline(num_classes=args.num_classes)
    elif args.arch == "SNN":
        model = SNN_baseline(num_classes=args.num_classes)
    elif args.arch == "single_transformer_tabular_crate_large":
        model = SingleTransformerTabularCRATE_large(num_classes=args.num_classes)
    elif args.arch == "inter_attention_single_transformer_tabular_crate_large":
        model = InterAttentionSingleTransformerTabularCRATE_large(num_classes=args.num_classes)
    else:
        raise NotImplementedError

    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.task == "classification":
        # define loss function (criterion), optimizer, and learning rate scheduler
        if "tabular" in args.arch.lower():
            # criterion = StaticFocalLoss(gamma=2.0, size_average=True).to(device)
            criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smooth).to(device)
        else:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smooth).to(device)
    else:
        criterion = torch.nn.MSELoss().to(device)

    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(0.9, 0.999),
                                      weight_decay=args.weight_decay)
    elif args.optimizer == "Lion":
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    warmup_steps = 20
    lr_func = lambda step: min((step + 1) / (warmup_steps + 1e-8),
                               0.5 * (math.cos(step / args.epochs * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        if args.tabular:
            raise NotImplementedError
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        if args.tabular:
            dataset = T1SNPDataset(t1_data_path=os.path.join(args.data, "new_t1_data.csv"),
                                   snp_data_path=os.path.join(args.data, "SNPS/All-Patients-SNPs.csv"),
                                   cognitive_data_path=os.path.join(args.data, "new_cognitive_data-first.csv"),
                                   target=args.target)

            dataset = T1SNPDataset.prepare_train_val_sets(dataset, task=args.task, train_size=0.8, random_seed=43)
            train_dataset = dataset.train
            val_dataset = dataset.val



        else:
            traindir = os.path.join(args.data, 'train')
            valdir = os.path.join(args.data, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            transform_simple = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

            train_dataset = datasets.ImageFolder(
                traindir,
                transform_simple
            )

            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
    print(f"I am using {args.workers} worker")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            print('distributed loader')
            train_sampler.set_epoch(epoch)
        else:
            print('non-distributed loader')

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1, _, _, _, _ = validate(val_loader, model, criterion, args)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, model_name=args.arch)

    # load the best model and do the evaluation
    checkpoint = torch.load(f"checkpoints/{args.arch}_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    val_best_acc, val_best_loss, val_best_f1, val_best_precision, val_best_recall = (
        validate(val_loader, model, criterion, args))
    return val_best_acc, val_best_loss, val_best_f1, val_best_precision, val_best_recall


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (t1_features, snp_features, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        t1_features = t1_features.to(device, non_blocking=True)
        snp_features = snp_features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with autocast():
            output = model(t1_features, snp_features)
            if criterion.__class__.__name__ == "LabelSmoothingCrossEntropy":
                target = target.squeeze(1)
            loss = criterion(output, target)

        # measure accuracy and record loss
        if args.task == "classification":
            acc1 = accuracy(output, target, topk=(1,))
        else:
            acc1 = r2(output, target)
        losses.update(loss.item(), t1_features.size(0))
        top1.update(acc1[0], t1_features.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (t1_features, snp_features, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    t1_features = t1_features.cuda(args.gpu, non_blocking=True)
                    snp_features = snp_features.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(t1_features, snp_features)
                if criterion.__class__.__name__ == "LabelSmoothingCrossEntropy":
                    target = target.squeeze(1)
                loss = criterion(output, target)

                # measure accuracy and record loss
                if args.task == "classification":
                    acc1 = accuracy(output, target, topk=(1,))
                    output = torch.argmax(output, dim=1)
                    f1_ = f1_score(output.cpu(), target.cpu(), average='macro')
                    precision_ = precision_score(output.cpu(), target.cpu(), average='macro')
                    recall_ = recall_score(output.cpu(), target.cpu(), average='macro')
                else:
                    acc1 = r2(output, target)
                    f1_, precision_, recall_ = 0, 0, 0

                losses.update(loss.item(), t1_features.size(0))
                top1.update(acc1[0], t1_features.size(0))
                f1.update(f1_, t1_features.size(0))
                precision.update(precision_, t1_features.size(0))
                recall.update(recall_, t1_features.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    f1 = AverageMeter('F1', ':6.2f', Summary.AVERAGE)
    precision = AverageMeter('Precision', ':6.2f', Summary.AVERAGE)
    recall = AverageMeter('Recall', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, f1, precision, recall],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg, losses.avg, f1.avg, precision.avg, recall.avg


def save_checkpoint(state, is_best, model_name='model'):
    save_dir = 'checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_file = f"{model_name}_checkpoint.pth.tar"
    best_file = f"{model_name}_best.pth.tar"
    torch.save(state, os.path.join(save_dir, checkpoint_file))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, checkpoint_file),
                        os.path.join(save_dir, best_file))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res


def r2(output, target):
    """Computes the R^2 value"""
    with torch.no_grad():
        return [r2_score(target.cpu().numpy(), output.cpu().numpy())]


if __name__ == '__main__':
    main()

# pos-embed +
# single-transformer +
