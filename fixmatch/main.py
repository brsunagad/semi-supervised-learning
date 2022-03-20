#some parts of code refered from:
#https://github.com/kekmodel/FixMatch-pytorch/blob/master/train.py

import argparse
import math
from torch.optim.lr_scheduler import LambdaLR

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy, evaluate, save_ckp, load_ckp

from model.wrn  import WideResNet
import torch.nn.functional as F
import torch.nn as N

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
import test

def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_sampler = RandomSampler
    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    sampler=train_sampler(labeled_dataset), 
                                    num_workers=args.num_workers,  drop_last=True))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch*args.mu,
                                    sampler=train_sampler(unlabeled_dataset),
                                    num_workers=args.num_workers,  drop_last=True))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    sampler=SequentialSampler(test_dataset),
                                    num_workers=args.num_workers,  drop_last=True)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)

############################################################################
# TODO: SUPPLY your code
############################################################################
    checkpoint_path = './savedmodels/new_fixmatchmodel_ckpt_{}_{}_{}.pt'.format(args.dataset, args.num_labeled, args.threshold, momentum = 0.9)
    best_model_path = './savedmodels/new_fixmatchmodel_{}_{}_{}.pt'.format(args.dataset, args.num_labeled, args.threshold)
    start_epoch = 0
    best_accuracy = 0
    print(args.threshold)
    print(args.num_labeled)
    print(args.dataset)
    print(args.lr)
    writer = SummaryWriter()
    model.train()
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=True)
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.wd)
    scheduler = get_cosine_schedule_with_warmup(
            optimizer, args.warmup, args.total_iter)
    if False:
      print('loading model')
      model, optimizer, start_epoch, best_accuracy = load_ckp(checkpoint_path, model, optimizer)
    optimizer.zero_grad()
    temp_threshold = args.threshold
    for epoch in range(start_epoch, args.epoch):
        for i in range(args.iter_per_epoch):
            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            sampler=train_sampler(labeled_dataset), 
                                            num_workers=args.num_workers, drop_last=True))
                x_l, y_l    = next(labeled_loader)
            
            try:
                (x_ul_weak, x_ul_strong), _     = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch*args.mu,
                                            sampler=train_sampler(unlabeled_dataset), 
                                            num_workers=args.num_workers, drop_last=True))
                (x_ul_weak, x_ul_strong), _    = next(unlabeled_loader)

            batch_size = x_l.shape[0]
            x = concatinate(
                torch.cat((x_l, x_ul_weak, x_ul_strong)), 2*args.mu+1).to(device)
            y_l = y_l.to(device)
            output = model(x)
            output = de_concatinate(output, 2*args.mu+1)
            output_x = output[:batch_size]
            output_u_weak, output_u_strong = output[batch_size:].chunk(2)
            del output
            labeled_loss = F.cross_entropy(output_x, y_l, reduction='mean')
            pseudo_label = torch.softmax(output_u_weak.detach(), dim=-1)
            scores, pseudo_targets = torch.max(pseudo_label, dim=-1)
            mask = scores.ge(args.threshold).float()
          
            unlabeled_loss = (F.cross_entropy(output_u_strong, pseudo_targets,
                                  reduction='none') * mask).mean()
            
            total_loss = labeled_loss + unlabeled_loss
            
            writer.add_scalar("Loss/train", total_loss, epoch)
            total_loss.backward()
            optimizer.step() 
            scheduler.step()
            optimizer.zero_grad()
        if (epoch+1) %1 == 0:
          test_acc, test_loss = evaluate(test_loader, len(test_dataset),model)
          print('Epoch: {} : Alpha Weight : {:.5f} | Test Acc : {:.5f} | Test Loss : {:.3f} '.format(epoch+1, alpha_weight(epoch), test_acc, test_loss))
          checkpoint = {'epoch': epoch+1, 'test_acc': test_acc, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),}
          save_ckp(checkpoint, False, checkpoint_path, best_model_path)
          writer.add_scalar("Accuracy/test", test_acc, epoch+1)
          if test_acc > best_accuracy:
            print('test accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_accuracy,test_acc))
            best_accuracy=test_acc
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
    test_model, _, _, _ = load_ckp(best_model_path, model, optimizer)
    accuracy, loss = evaluate(test_loader, len(test_dataset),test_model)
    writer.flush()
    writer.close()
    print('Test Acc : {:.5f} | Test Loss : {:.3f} '.format( accuracy, loss))
    if(args.num_classes == 10):
      print(test.test_cifar10(test_dataset, best_model_path).shape)
    elif(args.num_classes == 100):
      print(test.test_cifar100(test_dataset, best_model_path).shape)
############################################################################
#additional methods
############################################################################
def concatinate(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_concatinate(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def alpha_weight(epoch):
    T1 = 0
    T2 = 30
    af = 1
    if epoch < T1:
        return 0.0
    elif epoch > T2:
        return af
    else:
         return ((epoch-T1) / (T2-T1))*af

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
#################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.1, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=32, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=32, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=1024*100, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=34,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument('--mu', default=2, type=int,
                    help='coefficient of unlabeled batch size')
    parser.add_argument('--warmup', default=0, type=float,
                    help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)








