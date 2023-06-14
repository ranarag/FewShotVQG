import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet161, resnet152, vgg19

class IEncoder(nn.Module):
    def __init__(self):
        super(IEncoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  ~
#         # conv layer (depth from 16 --> 4), 3x3 kernels
#         self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
#         # pooling layer to reduce x-y dims by two; kernel and stride of 2
#         self.pool = nn.MaxPool2d(2, 2)
        
        self.net = resnet152(pretrained=True)
        self.net = nn.Sequential(*list(self.net.children())[:-2])
        self.dim = 2048
    def forward(self, x):

        x = self.net(x)
#         print(x.size())
        x = x.view(x.size(0), x.size(1), -1)
#         print(x.size())
#         exit()
        return x
        
        
class IDecoder(nn.Module):
    def __init__(self):
        super(IDecoder, self).__init__()
        ## decoder layers ##
        '''to be implemented'''
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, 2, 2), # 26 x 32
#             nn.ReLU(True),
# #             nn.Sigmoid(),
#             nn.BatchNorm2d(32),
            nn.ConvTranspose2d(2048, 256, 4, 4), # 256 x 28
            nn.ReLU(True),
#             nn.Sigmoid(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 64, 2, 2), # 64 x 56
            nn.ConvTranspose2d(64, 16, 2, 2), # 16 x 112
            nn.ConvTranspose2d(16, 3, 2, 2) # 3 x 224
            # nn.ReLU(True)
#             nn.Sigmoid()
            # nn.ConvTranspose2d(16, 3, 2, 2), # 224 x 3

        )
        
        
    def forward(self, x):
        return self.decoder(x)
 



