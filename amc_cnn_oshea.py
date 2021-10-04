#! /usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


class AMC_CNN_OShea(nn.Module):
    def __init__(
            self,
            num_output_classes,
            label_loss_object,
            domain_loss_object,
            learning_rate
        ):
        super(AMC_CNN_OShea, self).__init__()

        self.label_loss_object = label_loss_object
        self.domain_loss_object = domain_loss_object
        # self.conv = nn.Sequential(
        #     nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1),
        #     nn.ReLU(False),
        #     nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, stride=2),
        #     nn.ReLU(False),
        #     nn.Dropout(),
        # )
        nz = 100
        dim_domain = 1

        DROPOUT = 0.5
        NUM_CLASSES = 12
        self.net = nn.Sequential(
            nn.ZeroPad2d((2,2,1,1)), #left right top bottom
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1), nn.ReLU(True), nn.Dropout(DROPOUT),
            nn.ZeroPad2d((0,0,2,2)), #left right top bottom
            nn.Conv2d(in_channels=256, out_channels=80, kernel_size=3, stride=2), nn.ReLU(True), nn.Dropout(DROPOUT),
            nn.Flatten(),
        # )
        # self.net2 = nn.Sequential(
            nn.Linear(10240, 128), nn.ReLU(True),
            nn.Linear(128, 128), nn.ReLU(True),
            nn.Linear(128, 128), nn.ReLU(True),
            nn.Linear(128, NUM_CLASSES),
            nn.LogSoftmax(dim=1)
        )
        self.init_weight(self.net)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.lr_scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.5 ** (1 / 50))

    def init_weight(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)


    def forward(self, x, u):
        y_hat = self.net(x.reshape(-1, 1, 2, 128))

        return y_hat, torch.sum(y_hat, 1, keepdim=False)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def learn(self, x,y,u, alpha, domain_only:bool):
        """
        returns a dict of
        {
            label_loss:float, # if domain_only==False
            domain_loss:float
        }
        """

        # We are fudging the domain stuff
        y_hat, u_hat = self.forward(x,u) # Yeah it's dumb but I can't find an easy way to train the two nets separately without this
        loss = self.label_loss_object(y_hat, y)

        # TODO: Disable Encoder Learning
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "domain_loss": loss,
            "label_loss": loss
        }
