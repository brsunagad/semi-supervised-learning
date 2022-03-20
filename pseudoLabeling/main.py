#some parts of code refered from
#https://towardsdatascience.com/pseudo-labeling-to-deal-with-small-datasets-what-why-how-fd6f903213af
import argparse
import math

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy, test2_cifar10, save_ckp, load_ckp

from model.wrn  import WideResNet
import torch.nn.functional as F
import torch.nn as N

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader
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

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)

    ############################################################################
    # TODO: SUPPLY your code
    ############################################################################
    #print(args.wd)
    checkpoint_path = './savedmodels/model_ckpt_{}_{}_{}.pt'.format(args.dataset, args.num_labeled, args.threshold, momentum = 0.9)
    best_model_path = './savedmodels/model_{}_{}_{}.pt'.format(args.dataset, args.num_labeled, args.threshold)
    optimizer = torch.optim.SGD( model.parameters(), lr = args.lr, weight_decay=args.wd)
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.999),eps=1e-08,
                          #weight_decay=args.wd)
    model.train()
    #step = 20 
    clip_value = 5
    soft_max = N.Softmax(dim=1)
    criterion = N.CrossEntropyLoss()
    start_epoch = 0
    best_accuracy = 0
    print(args.threshold)
    print(args.num_labeled)
    print(args.dataset)
    if True:
      print('loading model')
      model, optimizer, start_epoch, best_accuracy = load_ckp(checkpoint_path, model, optimizer)
    for epoch in range(start_epoch, args.epoch):
        model.train()
        for i in range(args.iter_per_epoch):
            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)
            x_l, y_l    = x_l.to(device), y_l.to(device)

            if(epoch<20):
              output = model(x_l)
              labeled_loss = criterion(output, y_l)
              optimizer.zero_grad()
              labeled_loss.backward()
              optimizer.step()
            else:
              try:
                x_ul, _     = next(unlabeled_loader)
              except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch,
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_ul, _     = next(unlabeled_loader)
              x_ul        = x_ul.to(device)
              model.eval()
              output_ul = soft_max(model(x_ul))
              pseudo_scores, pseudo_labeled = torch.max(output_ul, 1)
             
              model.train()
              output = model(x_ul[pseudo_scores>args.threshold])
              _, pseudo_labeled = torch.max(output, 1)
              unlabeled_loss = alpha_weight(epoch) * criterion(output, pseudo_labeled)  
              output = model(x_l)
              labeled_loss = criterion(output, y_l)
              loss = labeled_loss + unlabeled_loss
              optimizer.zero_grad()
              loss.backward()
              #torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
              optimizer.step() 

        if (epoch+1) %5 == 0:
          test_acc, test_loss = test2_cifar10(test_loader, len(test_dataset),model)
          print('Epoch: {} : Alpha Weight : {:.5f} | Test Acc : {:.5f} | Test Loss : {:.3f} '.format(epoch+1, alpha_weight(epoch), test_acc, test_loss))
          checkpoint = {'epoch': epoch+1, 'test_acc': test_acc, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),}
          save_ckp(checkpoint, False, checkpoint_path, best_model_path)
          if test_acc > best_accuracy:
            print('test accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_accuracy,test_acc))
            best_accuracy=test_acc
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
          #model.train()
    #PATH = 'model_{}_{}_{}.pth.tar'.format(args.dataset, args.num_labeled, args.threshold)
    #torch.save(model, PATH)
    #test_output = test.test_cifar10(test_dataset, PATH)
    ########### test
    test_model, _, _, _ = load_ckp(best_model_path, model, optimizer)
    accuracy, loss = test2_cifar10(test_loader, len(test_dataset),test_model)
    print('Test Acc : {:.5f} | Test Loss : {:.3f} '.format( accuracy, loss))
            ####################################################################
            # TODO: SUPPLY your code
            ####################################################################
            
def alpha_weight(epoch):
    T1 = 19
    T2 = 40
    af = 3
    if epoch < T1:
        return 0.0
    elif epoch > T2:
        return af
    else:
         return ((epoch-T1) / (T2-T1))*af


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
    parser.add_argument('--total-iter', default=1600*70, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1600, type=int,
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
    
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)