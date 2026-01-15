import torch.nn as nn
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
import torch

class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v6', 
                           if_devide_out=True, use_norm=True, d_conv=2)

    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        output = self.block(input)
        return output + skip


class AttentionPropagation(nn.Module):
    def __init__(self, channels, head):
        nn.Module.__init__(self)
        self.head = head
        self.head_dim = channels // head

        self.query_filter = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.key_filter = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.value_filter0 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.value_filter1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        self.mh_filter0 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.mh_filter1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        x1 = x1.transpose(-1, -2)  # bcn
        x2 = x2.transpose(-1, -2)  # bcm

        query, key, value0, value1 = self.query_filter(x1).view(batch_size, self.head, self.head_dim, -1),\
                            self.key_filter(x2).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter0(x2).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter1(x1).view(batch_size, self.head, self.head_dim, -1)

        QK = torch.einsum('bhdn,bhdm->bhnm', query, key)
        score0 = torch.softmax(QK / self.head_dim ** 0.5, dim=-1)  # BHNM
        score1 = torch.softmax(QK / self.head_dim ** 0.5, dim=-2)  # BHNM
        # x1(q) attend to x2(k,v)
        add_value0 = torch.einsum('bhnm,bhdm->bhdn', score0, value0).reshape(batch_size, self.head_dim * self.head, -1)
        output0 = self.mh_filter0(add_value0)
        # x2(q) attend to x1(k,v)
        add_value1 = torch.einsum('bhnm,bhdn->bhdm', score1, value1).reshape(batch_size, self.head_dim * self.head, -1)
        output1 = self.mh_filter1(add_value1)

        return output0.transpose(-1, -2), output1.transpose(-1, -2)


class CrossMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.block0 = Mamba(dim, expand=1, d_state=8, bimamba_type='v2',
                           if_devide_out=True, use_norm=True, d_conv=2)
        self.block1 = Mamba(dim, expand=1, d_state=8, bimamba_type='v2',
                             if_devide_out=True, use_norm=True, d_conv=2)

        self.cat_filter0 = nn.Linear(dim, dim, bias=False)
        self.cat_filter1 = nn.Linear(dim, dim, bias=False)
        # cross
        self.cross = AttentionPropagation(dim, 4)

        # mapping
        self.map = nn.Linear(dim, 512, bias=False)  # BNM
        self.unmap = nn.Linear(dim, 512, bias=False)  # BNM
        self.proj_out = nn.Linear(dim, dim, bias=True)

    def forward(self, input):
        # input0: (B, N, C)
        B, N, C = input.shape

        skip = input
        input0 = self.norm0(input)
        # mamba 1
        skip0 = input0
        output0 = self.block0(input0)   # BNC

        # assign martix
        embed = self.map(input0)  # BNM
        _, _, M = embed.shape  # BNM
        S_map = torch.softmax(embed, dim=1)  # BNM
        # assign
        input1 = torch.matmul(input0.transpose(-1, -2), S_map).transpose(-1, -2)  # BCN * BNM -> BCM -> BMC
        skip1 = input1
        # mamba 2
        output1 = self.block1(input1)   # BMC

        # filter
        output0 = self.cat_filter0(output0) + skip0
        output1 = self.cat_filter1(output1) + skip1

        # cross attention
        output0_, output1_ = self.cross(output0, output1)
        output0 = output0 + output0_
        output1 = output1 + output1_

        # unmapping
        embed1 = self.unmap(input0)  # BNM
        S_unmap = torch.softmax(embed1, dim=2)  # BNM
        output1 = torch.matmul(output1.transpose(-1, -2), S_unmap.transpose(-1, -2)).transpose(-1, -2)  # BCM * BMN -> BCN -> BNC

        output = self.proj_out((output0 + output1)/2)
        return output + skip



class ConvFFN(nn.Module):

    def __init__(self, in_channels, hidden_channels,
                 out_channels, act_layer=nn.GELU, drop_out=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.fc3 = nn.Linear(hidden_channels, out_channels)
        inter_channels = int(hidden_channels // 4)

        self.drop = nn.Dropout(drop_out)
        self.act = act_layer()

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # b*128*1*1
            nn.Conv2d(hidden_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x: (b n c)
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x2 = x.transpose(-1, -2).unsqueeze(-1)
        xlg = self.global_att(x2)
        weight = self.sigmoid(xlg)

        x2 = self.fc2(x2*weight)
        x2 = self.act(x2)
        x2 = self.drop(x2)

        x2 = x2.squeeze(-1).transpose(-1, -2)

        x = self.fc3(x2) + x
        x = self.act(x)
        x = self.drop(x)

        return x

class Mamba_layer(nn.Module):
    def __init__(self, dim, depth=1):
        super().__init__()
        self.depth = depth
        self.mamba_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])
        for _ in range(depth):
            self.mamba_layers.append(CrossMambaBlock(dim))
            self.norm_layers.append(nn.LayerNorm(dim))
            self.ffn_layers.append(ConvFFN(dim, dim, dim, drop_out=0.))

    def forward(self, x):
        b, c, n, _ = x.shape
        pan = rearrange(x, 'b c h w -> b (h w) c')  # bnc
        for i in range(self.depth):
            pan = self.mamba_layers[i](pan)
            pan = self.ffn_layers[i](self.norm_layers[i](pan)) + pan
        pan = rearrange(pan, 'b (h w) c -> b c h w', h=n, w=1)
        return pan

