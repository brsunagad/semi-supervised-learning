Here is the command to run the main file in colab
!python -Wignore main.py --num-workers 8 --threshold 0.95 --num-labeled 2500 --dataset 'cifar100' --datapath './data/'
you can change the threshold, num-labeled and dataset to run different models
Some default parameters:
learning rate 0.1
Network depth 34
Network width 2
Total iterations 1600*70
Iterations per epoch 1600

saved models can be found in ./savedmodels
paths to load
model_cifar10_250_0.95.pt //model_{dataset}_{num_labeled}_{threshold}.pt
model_cifar10_250_0.75.pt
model_cifar10_250_0.6.pt
model_cifar10_4000_0.95.pt
model_cifar10_4000_0.75.pt
model_cifar10_4000_0.6.pt
model_cifar100_2500_0.95.pt 
model_cifar100_2500_0.75.pt
model_cifar100_2500_0.6.pt
model_cifar100_10000_0.95.pt
model_cifar100_10000_0.75.pt
model_cifar100_10000_0.6.pt