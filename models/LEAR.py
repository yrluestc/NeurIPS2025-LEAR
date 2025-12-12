# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import os
from datasets import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from backbone.utils.hsic import hsic
import torch.nn.functional as F
import random
import copy
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal


class LEAR(ContinualModel):
    NAME = 'LEAR'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(LEAR, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.train_loader_size = None
        self.iter = 0

    def end_task(self, dataset) -> None:
        #calculate distribution
        train_loader = dataset.train_loader
        num_choose = 100
        with torch.no_grad():
            train_iter = iter(train_loader)

            pbar = tqdm(train_iter, total=num_choose,
                        desc=f"Calculate distribution for task {self.current_task + 1}",
                        disable=False, mininterval=0.5)

            fc_features_list = []

            count = 0
            while count < num_choose:
                try:
                    data = next(train_iter)
                except StopIteration:
                    break

                x = data[0]
                x = x.to(self.device)

                processX = self.net.vitProcess(x)

                features = self.net.local_vitmodel.patch_embed(processX)
                cls_token = self.net.local_vitmodel.cls_token.expand(features.shape[0], -1, -1)
                features = torch.cat((cls_token, features), dim=1)
                features = features + self.net.local_vitmodel.pos_embed

                # forward pass till -3
                for block in self.net.local_vitmodel.blocks[:-3]:
                    features = block(features)

                features = self.net.Forever_freezed_blocks(features)

                features = self.net.local_vitmodel.norm(features)

                class_token = features[:, 0, :]

                fc_features_list.append(self.net.fcArr[self.current_task](class_token))

                count += 1
                pbar.update()

            pbar.close()
            fc_features = torch.cat(fc_features_list, dim=0)  # [num*b,fc_size]
            mu = torch.mean(fc_features, dim=0)
            sigma = torch.cov(fc_features.T)
            self.net.distributions.append(MultivariateNormal(mu, sigma))

        #deal with grad and blocks
        for fc in self.net.fcArr:
            for param in fc.parameters():
                param.requires_grad = False

        for cls in self.net.classifierArr:
            for param in cls.parameters():
                param.requires_grad = False

        self.net.Freezed_local_blocks = copy.deepcopy(torch.nn.Sequential(*list(self.net.local_vitmodel.blocks[-3:])))
        self.net.Freezed_global_blocks = copy.deepcopy(torch.nn.Sequential(*list(self.net.global_vitmodel.blocks[-3:])))

        for block in self.net.Freezed_local_blocks:
            for param in block.parameters():
                param.requires_grad = False

        for block in self.net.Freezed_global_blocks:
            for param in block.parameters():
                param.requires_grad = False

    def begin_task(self, dataset, threshold=0) -> None:
        train_loader = dataset.train_loader
        if self.current_task > 0:
            num_choose = 50
            with torch.no_grad():
                train_iter = iter(train_loader)

                pbar = tqdm(train_iter, total=num_choose,
                            desc=f"Choose params for task {self.current_task + 1}",
                            disable=False, mininterval=0.5)

                count = 0
                while count < num_choose:
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        break

                    x = data[0]
                    x = x.to(self.device)

                    processX = self.net.vitProcess(x)

                    features = self.net.local_vitmodel.patch_embed(processX)
                    cls_token = self.net.local_vitmodel.cls_token.expand(features.shape[0], -1, -1)
                    features = torch.cat((cls_token, features), dim=1)
                    features = features + self.net.local_vitmodel.pos_embed

                    # forward pass till -3
                    for block in self.net.local_vitmodel.blocks[:-3]:
                        features = block(features)

                    features = self.net.Forever_freezed_blocks(features)

                    features = self.net.local_vitmodel.norm(features)

                    class_token = features[:, 0, :]
                    distances = [0] * len(self.net.fcArr)
                    for t, (fc, dist) in enumerate(zip(self.net.fcArr, self.net.distributions)):
                        fc_feature = fc(class_token)
                        delta = fc_feature - dist.mean
                        inv_cov = torch.linalg.inv(dist.covariance_matrix)
                        mahalanobis = torch.sqrt(delta @ inv_cov @ delta.T).diagonal()
                        distances[t] += mahalanobis.mean()

                    count += 1
                    bar_log = {'distances': [round((x / count).item(), 2) for x in distances]}
                    pbar.set_postfix(bar_log, refresh=False)
                    pbar.update()
                pbar.close()

                min_idx = torch.argmin(torch.tensor(distances)).item()
                self.net.CreateNewExper(min_idx, dataset.N_CLASSES)

        self.opt = self.get_optimizer()

    def myPrediction(self,x,k):
        with torch.no_grad():
            #Perform the prediction according to the seloeced expert
            out = self.net.myprediction(x,k)
        return out

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        l2_distance = torch.nn.MSELoss()

        self.opt.zero_grad()
        if len(self.net.fcArr) > 1:
            outputs, Freezed_global_features, Freezed_local_features, global_features, local_features = self.net(inputs, return_features=True)
            loss_kd = kl_loss(local_features, Freezed_local_features)
            loss_mi = l2_distance(global_features, Freezed_global_features) #Directly calculate the L2 distance between features is more efficient than calculate MI between prediction, and it's also effective
            loss_hsic = -hsic(global_features, local_features)
            loss_ce = self.loss(outputs, labels)
            loss_tot = loss_ce + loss_kd + loss_hsic + loss_mi
            loss_vis = [loss_ce.item(), loss_kd.item(), loss_hsic.item(), loss_mi.item()]
        else:
            outputs, global_features, local_features = self.net(inputs)
            loss_hsic = -hsic(global_features, local_features)
            loss_ce = self.loss(outputs, labels)
            loss_tot = loss_ce + loss_hsic
            loss_vis = [loss_ce.item(), loss_hsic.item()]

        loss_tot.backward()


        self.opt.step()

        return loss_vis

    def cal_expert_dist(self,x):
        processX = self.net.vitProcess(x)

        features = self.net.local_vitmodel.patch_embed(processX)
        cls_token = self.net.local_vitmodel.cls_token.expand(features.shape[0], -1, -1)
        features = torch.cat((cls_token, features), dim=1)
        features = features + self.net.local_vitmodel.pos_embed

        # forward pass till -3
        for block in self.net.local_vitmodel.blocks[:-3]:
            features = block(features)

        features = self.net.Forever_freezed_blocks(features)

        features = self.net.local_vitmodel.norm(features)

        class_token = features[:, 0, :]
        distances = [0] * len(self.net.fcArr)
        for t, (fc, dist) in enumerate(zip(self.net.fcArr, self.net.distributions)):
            fc_feature = fc(class_token)
            delta = fc_feature - dist.mean
            inv_cov = torch.linalg.inv(dist.covariance_matrix)
            mahalanobis = torch.sqrt(delta @ inv_cov @ delta.T).diagonal()
            distances[t] += mahalanobis.mean().item()
        return distances



def kl_loss(student_feat, teacher_feat):
    student_feat = F.normalize(student_feat, p=2, dim=1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=1)

    student_prob = (student_feat + 1) / 2
    teacher_prob = (teacher_feat.detach() + 1) / 2

    loss_kld = F.kl_div(
        torch.log(student_prob + 1e-10),
        teacher_prob,
        reduction='batchmean'
    )
    return loss_kld