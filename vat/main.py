#Some parts of the code were referenced from the below links
#https://github.com/lyakaap/VAT-pytorch
#https://github.com/9310gaurav/virtual-adversarial-training
import argparse
import math

from dataloader import get_cifar10, get_cifar100
from vat        import VATLoss
from utils      import accuracy,evaluate, save_ckp, load_ckp
from model.wrn  import WideResNet
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data   import DataLoader
from torch.utils.tensorboard import SummaryWriter
import utils
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
    checkpoint_path = './savedmodels/model_ckpt_{}_{}.pt'.format(args.dataset, args.num_labeled)
    best_model_path = './savedmodels/model_{}_{}.pt'.format(args.dataset, args.num_labeled)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.999),eps=1e-08, weight_decay=args.wd)
    best_accuracy=0
    print("***************************************")
    print("Number of labelled: ")
    print(args.num_labeled)
    print("Dataset: ")
    print(args.dataset)
    print("Epsilon: ")
    print(args.vat_eps)
    print("Xi value: ")
    print(args.vat_xi)
    print("Learning rate: ")
    print(args.lr)
    print("***************************************")
    writer = SummaryWriter()
    for name, param in model.named_parameters():
        print(name, param.size())
    print(model.bn1.weight.data)
    return
    if False:
      print('loading model')
      model, optimizer, start_epoch, best_accuracy = load_ckp(checkpoint_path, model, optimizer)
    for epoch in range(args.epoch):
        accuracy = 0
        for i in range(args.iter_per_epoch):
            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)
            
            try:
                x_ul, _     = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch,
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_ul, _     = next(unlabeled_loader)
            
            x_l, y_l    = x_l.to(device), y_l.to(device)
            x_ul        = x_ul.to(device)

            optimizer.zero_grad()

            vat_loss = VATLoss(args)
            cross_entropy = nn.CrossEntropyLoss()
            lds = vat_loss(model, x_ul)
            output = model(x_l)
            classification_loss = cross_entropy(output, y_l)
            loss = classification_loss + args.alpha * lds
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()

            #acc = utils.evaluate(output, y_l)
        if (epoch+1) %5 == 0:
            test_acc, test_loss = evaluate(test_loader, len(test_dataset),model)
            print('Epoch: {} | Test Acc : {:.5f} | Test Loss : {:.3f} '.format(epoch+1, test_acc, test_loss))
            checkpoint = {'epoch': epoch+1, 'test_acc': test_acc, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),}
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)
            writer.add_scalar("Accuracy/test", test_acc, epoch+1)
            if test_acc > best_accuracy:
              print('test accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_accuracy,test_acc))
              best_accuracy=test_acc
              save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            ####################################################################
            # TODO: SUPPLY you code
            ####################################################################
    test_model, _, _, _ = load_ckp(best_model_path, model, optimizer)
    accuracy, loss = evaluate(test_loader, len(test_dataset),test_model)
    writer.flush()
    writer.close()
    print('Test Acc : {:.5f} | Test Loss : {:.3f} '.format( accuracy, loss))   
    if(args.num_classes == 10):
      print(test.test_cifar10(test_dataset, best_model_path).shape)
    elif(args.num_classes == 100):
      print(test.test_cifar100(test_dataset, best_model_path).shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.01, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=32, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=1600*50, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1600, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")                        
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=34,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument("--vat-xi", default=0.7, type=float, 
                        help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=3.0, type=float, 
                        help="VAT epsilon parameter") 
    parser.add_argument("--vat-iter", default=1, type=int, 
                        help="VAT iteration parameter") 
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)