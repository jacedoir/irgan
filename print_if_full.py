import torch.nn as nn

class one_conv(nn.Module):
    def __init__(self):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(16, 3, 1, 1)
        
    def forward(self, x):
        out = self.conv(x)
        return out


def for_preprocessed_image(image):
    model = one_conv()
    out = model(image)
    return out

