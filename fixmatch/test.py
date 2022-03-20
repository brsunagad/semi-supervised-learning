import torch
from dataloader import get_cifar10, get_cifar100
from torch.utils.data   import DataLoader

from model.wrn import WideResNet
from utils import accuracy
import torch.nn as F

def test_cifar10(testdataset, filepath = "./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 10]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-10. Test this
        function with the testdataset returned by get_cifar10()
    '''
    # TODO: SUPPLY the code for this function
    test_loader         = DataLoader(testdataset,
                                    batch_size = 64,
                                    shuffle = False, 
                                    num_workers=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = WideResNet(34, 10, widen_factor=2)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    m = F.Softmax(dim=1)
    temp = None
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            output = m(model(data))
            if(temp==None):
              temp = output
            else:
              temp = torch.cat((temp, output),0)
           
    return temp

def test_cifar100(testdataset, filepath="./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 100]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-100. Test this
        function with the testdataset returned by get_cifar100()
    '''
    # TODO: SUPPLY the code for this function
    test_loader         = DataLoader(testdataset,
                                    batch_size = 64,
                                    shuffle = False, 
                                    num_workers=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = WideResNet(34, 100, widen_factor=2)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    m = F.Softmax(dim=1)
    temp = None
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            output = m(model(data))
            if(temp==None):
              temp = output
            else:
              temp = torch.cat((temp, output),0)
           
    return temp


    