import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4
PT_D_MODEL = 256
PT_BACKBONE = "convnext_base"
PT_DROPOUT = 0.20

# Shared blocks
class MSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.1):
        super().__init__()
        self.conv3 = nn.Conv2d(in_ch, out_ch//2, 3, padding=1)
        self.conv5 = nn.Conv2d(in_ch, out_ch//4, 5, padding=2)
        self.conv7 = nn.Conv2d(in_ch, out_ch - (out_ch//2 + out_ch//4), 7, padding=3)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch//8, 1), nn.ReLU(True),
            nn.Conv2d(out_ch//8, out_ch, 1), nn.Sigmoid()
        )
        self.drop = nn.Dropout2d(drop)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        x = self.act(self.bn(x))
        x = x * self.se(x)
        x = self.drop(x)
        return self.pool(x)

class CNNPath(nn.Module):
    def __init__(self, in_ch=3, dims=(64,128,192), out_dim=PT_D_MODEL):
        super().__init__()
        c = in_ch
        blocks=[]
        for d in dims:
            blocks.append(MSConvBlock(c, d))
            c = d
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, out_dim)
        )

    def forward(self, x):
        return self.head(self.blocks(x))

class CPAF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(True),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )
        self.out = nn.Linear(dim*2, dim)

    def forward(self, f_cnn, f_glob):
        w = self.gate(torch.cat([f_cnn, f_glob], dim=-1))
        f = torch.stack([f_cnn, f_glob], dim=1)
        fused = (w.unsqueeze(-1) * f).sum(dim=1)
        return self.out(torch.cat([fused, fused], dim=-1))

class ViewAttention(nn.Module):
    def __init__(self, dim, views=3):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(views))

    def forward(self, feats):
        w = F.softmax(self.alpha, dim=0).view(1,-1,1)
        return (feats * w).sum(dim=1)

class GlobalPathPretrained(nn.Module):
    def __init__(self, in_ch, d_model, backbone='convnext_base'):
        super().__init__()
        self.bb = timm.create_model(backbone, pretrained=True, in_chans=in_ch, num_classes=0, global_pool="avg")
        self.proj = nn.Linear(self.bb.num_features, d_model)

    def forward(self, x):
        return self.proj(self.bb(x))

class MaViCNetPT(nn.Module):
    def __init__(self, num_classes, d_model, backbone='convnext_base', head_dropout=0.2):
        super().__init__()
        self.cnn  = CNNPath(in_ch=3, dims=(64,128,192), out_dim=d_model)
        self.glob = GlobalPathPretrained(3, d_model, backbone)
        self.cpaf = CPAF(d_model)
        self.view = ViewAttention(d_model, views=3)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.ReLU(True),
            nn.Dropout(head_dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, mv):
        feats=[]
        for v in range(mv.size(1)):
            x = mv[:,v]
            feats.append(self.cpaf(self.cnn(x), self.glob(x)))
        g = self.view(torch.stack(feats, dim=1))
        return self.head(g)

