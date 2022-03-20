This folder is for our implementation of just fixmatch to compare against our changes to fixmatch which are present in another folder named task3
Here is the command to run the main file in colab
!python -Wignore main.py --threshold 0.95 --num-labeled 10000 --dataset 'cifar100' --lr 0.001 --datapath './data/'
you can change the threshold, num-labeled and dataset to run different models
Some default parameters:
learning rate 0.1
Network depth 34
Network width 2
Total iterations 1024*100
Iterations per epoch 1024

trained models are saved in savedmodels folder
paths for different models:
./savedmodels/fixmatchmodel_cifar10_250_0.95.pt
./savedmodels/fixmatchmodel_cifar10_4000_0.95.pt
./savedmodels/new_fixmatchmodel_cifar100_2500_0.95.pt
./savedmodels/new_fixmatchmodel_cifar100_10000_0.95.pt
