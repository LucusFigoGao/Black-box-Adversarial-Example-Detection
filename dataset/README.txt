Adversarial examples save in this folder

1). [training adversarial examples]
    normal-resnet18-cifar, normal-resnet18-cifar100

2). [SM-AT adversarial examples]
    adver-resnet18-cifar, adver-resnet18-cifar100

3). [SM-NT adversarial examples]
    normal-vgg16-cifar, normal-vgg16-cifar100
    normal-wideresnet28-cifar, normal-wideresnet28-cifar100

4). [EM-AT adversarial examples]
    adver-ensemble-cifar, adver-ensemble-cifar100

5). [EM-NT adversarial examples]
    normal-ensemble-cifar, normal-ensemble-cifar100


each folder contains six subfolders: 1) —— 3)
    FGSM,             BIM,              PGD-l2,          PGD-linf,      DeepFool,        C&W
    -AE-0.01.npy      -AE-0.01.npy      -AE-1.0.npy      -AE-8.npy      -AE-0.5.npy      -AE-0.5.npy
    -NE-0.01.npy      -NE-0.01.npy      -NE-1.0.npy      -NE-8.npy      -NE-0.5.npy      -NE-0.5.npy
    -CE-0.01.npy      -CE-0.01.npy      -CE-1.0.npy      -CE-8.npy      -CE-0.5.npy      -CE-0.5.npy
    -label-0.01.npy   -label-0.01.npy   -label-1.0.npy   -label-8.npy   -label-0.5.npy   -label-0.5.npy

each folder contains four files: 4) —— 5)
    AE-8.npy, NE-8.npy, CE-8.npy, label-8.npy