import torch
import torch.nn as F
import torch.nn.functional as N
import shutil

def accuracy(output, target, topk=(1,)):
    """
    Function taken from pytorch examples:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified
    values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def test2_cifar10(test_loader, len_dataset, model):
    correct = 0 
    loss = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = torch.load(filepath)
    model.to(device)
    model.eval()
    m = F.Softmax(dim=1)
    criterion = F.CrossEntropyLoss()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            output = m(model(data))
            predicted = torch.max(output,1)[1]
            correct += (predicted == labels.cuda()).sum()
            predicted = torch.max(output,1)[1]
            loss += criterion(output, labels.cuda()).item()
    accuracy = (float(correct)/len_dataset) *100
    loss = (loss/len(test_loader))
    return accuracy, loss

def save_ckp(state, is_best, checkpoint_path, best_model_path):
#https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['test_acc']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min   


