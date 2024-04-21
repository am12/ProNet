from torch.nn import Module, BatchNorm1d, LeakyReLU, Conv1d, ModuleList, Softmax, Linear
import numpy as np

CARDINALITY_ITEM = 16
INPUT_CHANNELS = 20
NUM_CLASSES = 25
SEQUENCE_LEN = 600

class ResidualUnit(Module):
    def __init__(self, l, w, ar, bot_mul=1):
        super().__init__()
        bot_channels = int(round(l * bot_mul))
        self.batchnorm1 = BatchNorm1d(l)
        self.relu = LeakyReLU(0.1)
        self.batchnorm2 = BatchNorm1d(l)
        self.C = bot_channels // CARDINALITY_ITEM
        self.conv1 = Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2, groups=self.C)
        self.conv2 = Conv1d(l, l, w, dilation=ar, padding=(w-1)*ar//2, groups=self.C)

    def forward(self, x, y):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x2 = self.relu(self.batchnorm2(self.conv2(x1)))
        return x + x2, y

class Skip(Module):
    def __init__(self, l):
        super().__init__()
        self.conv = Conv1d(l, l, 1)

    def forward(self, x, y):
        return x, self.conv(x) + y

class ProNet(Module):
    def __init__(self, L=64, W=np.array([11]*8+[21]*4+[41]*4), AR=np.array([1]*4+[4]*4+[10]*4+[25]*4)):
        super().__init__()
        self.conv1 = Conv1d(INPUT_CHANNELS, L, 1) 
        self.skip1 = Skip(L)
        self.residual_blocks = ModuleList()
        for i, (w, r) in enumerate(zip(W, AR)):
            self.residual_blocks.append(ResidualUnit(L, w, r))
            if (i+1) % 4 == 0:
                self.residual_blocks.append(Skip(L))
        if (len(W)+1) % 4 != 0:
            self.residual_blocks.append(Skip(L))
        self.last_cov = Conv1d(L, L, 1)
        self.final_layer = Linear(L*SEQUENCE_LEN, NUM_CLASSES)  # change here for 25 classes
        # self.softmax = Softmax(dim=1)  # softmax along the feature dimension

    def forward(self, x):
        x, skip = self.skip1(self.conv1(x), 0)
        for m in self.residual_blocks:
            x, skip = m(x, skip)
        x = self.last_cov(skip)
        x = x.view(x.size(0), -1)  # Flatten the output for the linear layer
        x = self.final_layer(x)
        return x # cross entropy loss expects raw logits