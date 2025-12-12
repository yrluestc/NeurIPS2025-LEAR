# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

import timm
import torchvision.transforms as transforms
import copy

from backbone import MammothBackbone, register_backbone



class LEAR(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Instantiates the layers of the network.
        """

        super(LEAR, self).__init__()

        self.device = 'cuda:0'

        self.num_classes = num_classes

        self.model_dim = 768
        self.fc_dim = 500

        self.fcArr = [nn.Linear(self.model_dim, self.fc_dim, device=self.device)]
        self.classifierArr = [nn.Linear(self.model_dim + self.fc_dim, self.num_classes, device = self.device)]
        self.distributions = []

        model_name_vit = 'vit_base_patch16_224'

        self.c_expert = 0

        self.vitProcess = transforms.Compose(
        [transforms.Resize(224)])

        self.local_vitmodel = timm.create_model(
            model_name_vit,
            pretrained=True,
            num_classes=self.num_classes
        )

        self.global_vitmodel = timm.create_model(
            model_name_vit,
            pretrained=True,
            num_classes=self.num_classes
        )

        for param in self.local_vitmodel.parameters():
            param.requires_grad = False

        for param in self.global_vitmodel.parameters():
            param.requires_grad = False

        # partially activated
        num_unfrozen_layers = 3
        for block in self.local_vitmodel.blocks[-num_unfrozen_layers:]:
            for param in block.parameters():
                param.requires_grad = True

        for block in self.global_vitmodel.blocks[-num_unfrozen_layers:]:
            for param in block.parameters():
                param.requires_grad = True

        self.Freezed_global_blocks = None
        self.Freezed_local_blocks = None

        self.Forever_freezed_blocks = copy.deepcopy(torch.nn.Sequential(*list(self.local_vitmodel.blocks[-3:])))
        for block in self.Forever_freezed_blocks:
            for param in block.parameters():
                param.requires_grad = False

    


    def CreateNewExper(self, idx, num_classes):
        new_fc_dim = self.fc_dim
        new_fc = nn.Linear(self.model_dim, new_fc_dim, device=self.device)
        new_fc.load_state_dict(self.fcArr[idx].state_dict())
        # self.classifier.load_state_dict(self.classifierArr[idx].state_dict())
        print('load expert ' + str(idx+1) + ' parameters')
        self.fcArr.append(new_fc)
        self.classifier = nn.Linear(self.model_dim + new_fc_dim, num_classes, device=self.device)
        self.classifierArr.append(self.classifier)

        print('Create new expert ' + str(len(self.classifierArr)))

        self.c_expert = len(self.classifierArr) - 1

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def forward_expert(self, global_features, local_features):
        fcfeatures = self.fcArr[self.c_expert](local_features)
        final_features = torch.cat((global_features,fcfeatures), dim=1)
        out = self.classifierArr[self.c_expert](final_features)
        return out

    def forward(self, x: torch.Tensor, return_features=False) -> torch.Tensor:
        if return_features:
            Freezed_global_features, Freezed_local_features, global_features, local_features = self.forward_fusion(x, return_features=True)
            return self.forward_expert(global_features, local_features), Freezed_global_features, Freezed_local_features, global_features, local_features
        else:
            global_features, local_features = self.forward_fusion(x)
            return self.forward_expert(global_features, local_features), global_features, local_features

    def forward_fusion(self, x, return_features=False):

        processX = self.vitProcess(x)

        local_features = self.local_vitmodel.patch_embed(processX)
        local_cls_token = self.local_vitmodel.cls_token.expand(local_features.shape[0], -1, -1)
        local_features = torch.cat((local_cls_token, local_features), dim=1)
        local_features = local_features + self.local_vitmodel.pos_embed

        global_features = self.global_vitmodel.patch_embed(processX)
        global_cls_token = self.global_vitmodel.cls_token.expand(global_features.shape[0], -1, -1)
        global_features = torch.cat((global_cls_token, global_features), dim=1)
        global_features = global_features + self.global_vitmodel.pos_embed

        # forward pass till -3 layer
        for block in self.local_vitmodel.blocks[:-3]:
            local_features = block(local_features)

        for block in self.global_vitmodel.blocks[:-3]:
            global_features = block(global_features)


        if return_features:
            Freezed_global_features = self.Freezed_global_blocks(global_features)
            Freezed_local_features = self.Freezed_local_blocks(local_features)

        for block in self.local_vitmodel.blocks[-3:]:
            local_features = block(local_features)

        for block in self.global_vitmodel.blocks[-3:]:
            global_features = block(global_features)

        local_features = self.local_vitmodel.norm(local_features)
        local_features = local_features[:, 0, :]
        global_features = self.global_vitmodel.norm(global_features)
        global_features = global_features[:, 0, :]

        if return_features:
            Freezed_global_features = self.global_vitmodel.norm(Freezed_global_features)
            Freezed_global_features = Freezed_global_features[:, 0, :]
            Freezed_local_features = self.local_vitmodel.norm(Freezed_local_features)
            Freezed_local_features = Freezed_local_features[:, 0, :]

            return Freezed_global_features, Freezed_local_features, global_features, local_features
        else:
            return global_features, local_features

    def myprediction(self,x,index):
        with torch.no_grad():
            global_features, local_features = self.forward_fusion(x)
            fcfeatures = self.fcArr[index](local_features)
            final_features = torch.cat((global_features, fcfeatures), dim=1)
            out = self.classifierArr[index](final_features)
            return out

@register_backbone("lear")
def LEAR_backbone(num_classes):
    return LEAR(num_classes)





