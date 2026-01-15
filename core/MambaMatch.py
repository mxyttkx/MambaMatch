import torch
import torch.nn as nn
from loss import batch_episym
from einops import rearrange
import numbers
import torch.nn.functional as F
from mamba_block import Mamba_layer as MSMamba


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)


class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),  # 4或6 → 128
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        ###################################################################
        self.embed_0 = MSMamba(dim=self.out_channel, depth=2)
        self.embed_1 = MSMamba(dim=self.out_channel, depth=2)
        ###################################################################
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))
        ####################################################################
        if self.predict:
            self.embed_2 =  ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)]  # in
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)  # y_o
            w_out = torch.gather(weights, dim=-1, index=indices)  # w_
        indices = indices.view(B, 1, -1, 1)  #

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))  # x
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))  #
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))  #
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        B, _, N, _ = x.size()
        out = x.transpose(1, 3).contiguous()  #
        out = self.conv(out)  # o
        #######################################################
        out = self.embed_0(out)  # o
        w0 = self.linear_0(out).view(B, -1)  # w
        ######################################################
        out = self.embed_1(out)
        w1 = self.linear_1(out).view(B, -1)  # w
        ##########################################################
        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            out = self.embed_2(out)
            w2 = self.linear_2(out)
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat


class MambaMatch_Net(nn.Module):
    def __init__(self, config):
        super(MambaMatch_Net, self).__init__()

        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=config.sr)  #
        self.ds_1 = DS_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr)

    def forward(self, x, y):
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0 = self.ds_0(x, y)  #

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)  #
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)  #

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1)  #

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)  #

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), 'U')
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    mask = logits[:, 0, :, 0]
    weights = logits[:, 1, :, 0]

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen

    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat




