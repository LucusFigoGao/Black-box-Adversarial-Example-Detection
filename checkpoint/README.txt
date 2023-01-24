checkpoints save in this file: (victim and threat model)

AT:             0/1 (adversarial training or not)
dataset:        cifar/cifar100
architecture:   resnet18/vgg16/wideresnet28
S:              666/777/888 (random seed)

RESUME_DEFAULT = {
    "normal-resnet18-cifar": "./checkpoint/AT:0-cifar-resnet18-S:888/checkpoint.pt.best", 
    "normal-resnet18-cifar100": "./checkpoint/AT:0-cifar100-resnet18-S:666/checkpoint.pt.best", 
    "adver-resnet18-cifar": "./checkpoint/AT:1-cifar-resnet18-S:888/checkpoint.pt.best", 
    "adver-resnet18-cifar100": "./checkpoint/AT:1-cifar100-resnet18-S:777/checkpoint.pt.best", 
    "normal-vgg16-cifar": "./checkpoint/AT:0-cifar-vgg16-S:777/checkpoint.pt.best", 
    "normal-vgg16-cifar100": "./checkpoint/AT:0-cifar100-vgg16-S:777/checkpoint.pt.best", 
    "normal-wideresnet28-cifar": './checkpoint/AT:0-cifar-wideresnet28-S:666/checkpoint.pt.best', 
    "normal-wideresnet28-cifar100": './checkpoint/AT:0-cifar100-wideresnet28-S:777/checkpoint.pt.best', 
}