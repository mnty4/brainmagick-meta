from torch import nn
import torch
from .belt_output import Belt3MaeClfOutput
from torch.nn import functional as F

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def __init__(self, 
                 n_classes=2, 
                 channels=60, 
                 time_window=151,
                 dropoutRate=0.5, 
                 kernelLength=64, 
                 kernelLength2=16, 
                 F1=16,
                 D=2, 
                 F2=16):
        super().__init__()
        print('EEGNet init')
        self.F1 = F1 # number of filter for channelwise convolution
        self.kernelLength  = kernelLength  # 
        self.F2 = F2
        self.D = D
        self.time_window = time_window
        self.n_classes = n_classes
        self.channels = channels
        
        self.kernelLength2 = kernelLength2
        self.dropoutRate = dropoutRate
        self.blocks = self.InitialBlocks(dropoutRate)
        self.blockOutputSize = self.CalculateOutSize(self.blocks, channels, time_window)
        self.classifierBlock = self.ClassifierBlock(self.F2 * self.blockOutputSize[1], n_classes)
        self.apply(self._init_weights)
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.block_weights = ['blocks.0.0.weight', 'blocks.0.1.weight', 'blocks.1.0.weight', 'blocks.1.1.weight','blocks.0.1.bias', 'blocks.1.2.bias']

    def InitialBlocks(self, dropoutRate, *args, **kwargs):
        block1 = nn.Sequential(
            # temporal convolution
            nn.Conv2d(1,self.F1,(1,self.kernelLength),stride=1,padding=(0,self.kernelLength//2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3), #norm
            # channel wise convolution
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
                                 groups=self.F1, bias=False),            
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropoutRate))
        
        block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            # ========================================

            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutRate))
        return nn.Sequential(block1, block2)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def ClassifierBlock(self, inputSize, n_classes):
        return nn.Sequential(
            nn.Linear(inputSize, n_classes, bias=False),
            nn.Softmax(dim=1))

    def CalculateOutSize(self, model, channels, time_window):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, channels, time_window)
        model.eval()
        out = model(data).shape
        # print('out',out)
        # exit(0)
        return out[2:]
    def embed(self, eeg):
        # print('EEGNet, embed',eeg.size())
        x = self.blocks(eeg)
        
        x = x.squeeze()
        # print('squeeze',x.size())
        return x
    def calculate_loss(self, logits, targets):
        loss = self.loss(logits, targets)
        return loss
    def forward(self, input_sample):
        x = input_sample['eeg'].float()
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        else:
            x = x.transpose(1, 2)
        # print('x',x.size())
        # expand dimension for channelwise convolution
        # x = x.unsqueeze(1)
        x = self.blocks(x)
        print(x.shape)
        # print('blocks',x.size()  )
        x = x.view(x.size()[0], -1)  # Flatten
        print(x.shape)
        x = self.classifierBlock(x)
        # softmax
        x = F.softmax(x, dim=1)
        loss = self.calculate_loss(x, input_sample['label'])
        return Belt3MaeClfOutput(
            loss=loss,
            # loss_mae=mae_loss,
            # loss_clf=10*clf_loss,
            logits=x
        )
    def forward2(self, input_sample):
        x = input_sample['eeg'].float()
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        else:
            x = x.transpose(1, 2)
        # print('x',x.size())
        # expand dimension for channelwise convolution
        # x = x.unsqueeze(1)
        x = self.blocks(x)
        # print('blocks',x.size()  )
        x = x.view(x.size()[0], -1)  # Flatten
        x = self.classifierBlock(x)
        # softmax
        x = F.softmax(x, dim=1)
        loss = self.calculate_loss(x, input_sample['label'])
        return loss
    def generate(self, input_sample):
        out  = self.forward(input_sample)
        # apply softmax
        out['logits'] = F.softmax(out['logits'], dim=1)
        
        output = {"loss_all": out['loss'],
                  "loss": out['loss'],
                  'raw_eeg': input_sample['eeg'],
                  'logits': out['logits'],
                  'label': input_sample['label']}
        return output
    def freeze(self):
        for name, param in self.blocks.named_parameters():
            param.requires_grad = False
    def unfreeze(self):
        for name, param in self.blocks.named_parameters():
            param.requires_grad = True


    def _load_weights(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        msg1 = self.load_state_dict(state_dict)
        if msg1.missing_keys:
            print("missing keys: {}".format(msg1.missing_keys))
        print("successfully load model from {}".format(path))