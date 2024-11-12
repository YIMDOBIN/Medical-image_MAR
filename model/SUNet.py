import torch.nn as nn
from model.SUNet_detail import SUNet


class SUNet_model(nn.Module):
    def __init__(self):
        super(SUNet_model, self).__init__()
        self.swin_unet = SUNet(img_size=512,
                                patch_size=4,
                                in_chans=1,
                                out_chans=1,
                                window_size=8,
                                upscale=1,
                                num_classes=1)

    def forward(self, x):
        #if x.size()[1] == 1:
        #    x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits
    
if __name__ == '__main__':
    from utils.model_utils import network_parameters
    import torch
    import yaml
    from thop import profile
    from utils.model_utils import network_parameters

