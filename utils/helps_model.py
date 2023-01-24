# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   helps_model.py
    Time:        2022/10/24 15:18:53
    Editor:      Figo
-----------------------------------
'''

import os
import torch
import torch.nn as nn


class ModuleHelpers(nn.Module):
    def __init__(self, model) -> None:
        super(ModuleHelpers, self).__init__()
        self.model = model
        self._check_model()
    
    def _check_model(self):
        for attribute in ['module', 'model']:
            if hasattr(self.model, attribute):
                self.model = getattr(self.model, attribute)

    def _get_layer(self, layer_name):
        for name, submodule in self.model.named_children():
            if name == layer_name:
                return submodule
        print(f"=> No module named {layer_name}")
        return None

    @ staticmethod
    def _modify_layer(super_module, module_name, new_module):
        """
            :: Build a new space for modifying layers operations.
            :: e.x.
            >>> MHelpers = ModuleHelpers(model)
            >>> ori_layer = MHelpers._get_layer("layer1")
            >>> MHelpers._modify_layer(ori_layer, "IBM-layer1", nn.ReLU())
        """
        if isinstance(super_module, nn.Sequential):
            super_module.add_module(module_name, nn.Sequential())
        if not isinstance(new_module, nn.Sequential):
            new_module = nn.Sequential(new_module)
        setattr(super_module, module_name, new_module)
        print(f"=> layer {module_name} has been modified...")


class GradientHook:
    def __init__(self, module, device="cuda:0") -> None:
        self.module, self.device = module, device
        self.input, self.output = None, None
        self.hook = self.module.register_forward_hook(self.__grad_hook)
    
    def remove_grad_hook(self):
        self.hook.remove()
    
    def __grad_hook(self, module, input, output):
        self.input = input
        self.output = output


class IndividualGradients:
    def __init__(self, model, device="cuda:0", layer_name='layer1') -> None:
        """
            :: Return feature map & gradient map from specific layers
            :: e.x.
            >>> model, dataset = easy_load_model_dataset_for_test('resnet18', 'cifar')
            >>> images = torch.randn((17, 3, 32, 32), dtype=torch.float32).cuda()
            >>> target = torch.randint(0, 10, (17, ), dtype=torch.long).cuda()
            >>> print("=> Model & dataset is OK...")

            >>> IG = IndividualGradients(model, layer_name='layer2')
            >>> features_maps = IG.get_individual_features(images)[0]               # get feature map
            >>> print("=> feature map shape is:", features_maps.shape)
            >>> => feature map shape is: torch.Size([17, 128, 16, 16])

            >>> outputs = model(images, with_latent=False, with_image=False)
            >>> gradinet_maps = IG.get_individual_gradient(outputs, target)[0]      # get gradient map
            >>> print("=> gradient map shape is:", gradinet_maps.shape)
            >>> => gradient map shape is: torch.Size([17, 128, 16, 16])
        """
        self.model, self.device = model, device
        self.helpers = ModuleHelpers(self.model)
        self.module = self.helpers._get_layer(layer_name)
        self.indiv_fea = [GradientHook(self.module, self.device)]
    
    def get_individual_gradient(self, outputs, targets, criterion=nn.CrossEntropyLoss()):
        loss = criterion(outputs, targets)
        gradients_layers = []
        for feature_maps in self.indiv_fea:
            gradients = torch.autograd.grad(loss, feature_maps.output, retain_graph=True)
            gradients = torch.squeeze(gradients[0])
            gradients_layers.append(gradients)
            feature_maps.remove_grad_hook()
        return gradients_layers

    def get_individual_features(self, inputs):
        outputs = self.model(inputs, with_image=False)
        features_layers = []
        for feature_maps in self.indiv_fea:
            features_layers.append(feature_maps.output)
            feature_maps.remove_grad_hook()
        
        return features_layers


def reload_model(model, replacement):
    """
        :: reload modified model into AttackerModel class
    """
    if hasattr(model, 'module'):
        model = getattr(model, 'module')
    setattr(model, 'model', replacement)
    return model


def frozen(model, layer=None, is_warm=False):
    """
        :: Frozen specific layers or the while model
        :: e.x.
        >>> model = frozen(model, 'linear', True)
    """
    net = model
    
    for attribute in ['module', 'model']:
        if hasattr(model, attribute):
            model = getattr(model, attribute)
    
    #! All layers will be frozen
    if layer is None:
        for name, params in model.named_parameters():
            params.requires_grad = False
    #! Specific layers will be frozen
    else:
        for name, submodule in model.named_children():
            for _, params in submodule.named_parameters():
                params.requires_grad = is_warm if name in layer else (not is_warm)
    print("=> Model parameters has been frozen...")
    return reload_model(net, model)


def is_requires_grad(module):
    for name, params in module.named_parameters():
        print(name, " requires grad:", params.requires_grad)


def load_random_parameters(model, layer):
    """
        :: Load random parameters at specific layer
        :: e.x.
        >>> model = load_random_parameters(model, "linear")
    """
    net = model
    
    for attribute in ['module', 'model']:
        if hasattr(model, attribute):
            model = getattr(model, attribute)
    
    for name, module in model.named_children():
        if name == layer:
            print(f"=> Locate {layer}...")
            for n, submodule in module.named_modules():
                if isinstance(submodule, nn.Conv2d):
                    print(f"=> Load convlutional layer {n}...")
                    nn.init.kaiming_normal_(submodule.weight, mode='fan_out', nonlinearity='relu')
                    if submodule.bias is not None:
                        nn.init.constant_(submodule.bias, 0)
                elif isinstance(submodule, nn.BatchNorm2d):
                    print(f"=> Load batchnorm layer {n}...")
                    nn.init.constant_(submodule.weight, 1)
                    nn.init.constant_(submodule.bias, 0)
                elif isinstance(submodule, nn.Linear):
                    print(f"=> Load linear layer {n}...")
                    nn.init.normal_(submodule.weight, 0, 0.01)
                    nn.init.constant_(submodule.bias, 0)
    return reload_model(net, model)


def get_parameters(model, layer=None):
    for attribute in ['module', 'model']:
        if hasattr(model, attribute):
            model = getattr(model, attribute)
    if layer is None:
        return list(model.parameters())
    update_params = []
    for name, submodule in model.named_children():
        if name == layer:
            for _, parameters in submodule.named_parameters():
                update_params.append(parameters)
    return update_params


def laod_pretrain_model(model, pth):
    if not os.path.isfile(pth):
        raise ValueError(f"=> No such files {pth}")
    cpt, fprint = torch.load(pth, map_location="cpu"), "=> all keys match..."
    if "model" in cpt.keys():
        if 'epoch' in cpt.keys():
            fprint += "Best params is {} epochs |".format(cpt['epoch'])
        if 'loss' in cpt.keys():
            fprint += "loss: {:.3f} |".format(cpt['loss'][-1])
        if 'acc' in cpt.keys():
            fprint += "accuracy: {:.3f} |".format(cpt['acc'])
        cpt = cpt['model']
    model.load_state_dict(cpt)
    print(fprint)
    return model


def get_mean_var(model):
    
    with torch.no_grad():
        model.eval()
        for attribute in ['module', 'model']:
            if hasattr(model, attribute):
                model = getattr(model, attribute)

        running_mean, running_var = {}, {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                running_mean[name] = module.running_mean.data
                running_var[name] = module.running_var.data
    
    return running_mean, running_var


if __name__ == "__main__":

    from helps_data import easy_load_model_dataset_for_test

    #! Load model and image
    model, dataset = easy_load_model_dataset_for_test('resnet18', 'cifar')
    images = torch.randn((17, 3, 32, 32), dtype=torch.float32).cuda()
    target = torch.randint(0, 10, (17, ), dtype=torch.long).cuda()
    print("=> Model & dataset is OK...")
    running_mean, running_var = get_mean_var(model)
    print(running_mean)
    