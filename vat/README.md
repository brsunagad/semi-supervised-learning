Here is the command to run the main file in colab
!python -Wignore main.py --num-workers  8 --dataset 'cifar10' --num-labeled 4000 --vat-xi 0.3 --vat-eps 7.0 --lr 0.03 --datapath './data/'
you can change the threshold, num-labeled,dataset, VAT Xi, VAT epsilon and learning rate to run different models
Some default parameters:
learning rate 0.03
Network depth 34
Network width 2
Total iterations 1600*50
Iterations per epoch 1600

trained models are saved in savedmodels folder
paths for different models:
./savedmodels/model_cifar10_250.pt
./savedmodels/model_cifar10_4000.pt
./savedmodels/model_cifar100_2500.pt
./savedmodels/model_cifar100_10000.pt