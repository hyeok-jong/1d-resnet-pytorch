import torch
import torch.nn as nn



class Residual_Block(nn.Module):
    '''
    This Residual Block basically maintain the lenght
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_layers, drop_p):
        super(Residual_Block, self).__init__()

        assert kernel_size % 2 ==1, f'kernel_size {kernel_size} should be odd'
        
        padding = (kernel_size - 1) // 2

            
        Conv_Blocks = list()
        for n in range(n_layers):
            if n != 0:
                in_channels = out_channels
            Conv_Blocks.append(
                self.BaseModule(in_channels, out_channels, kernel_size, padding, drop_p)
            )
        self.Conv_Blocks = nn.Sequential(*Conv_Blocks)

        self.ReLU = nn.ReLU()

    def BaseModule(self, in_channels, out_channels, kernel_size, padding, drop_p):
        Conv_Block = [
            nn.Conv1d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = 1,
                padding = padding
            ),
            nn.BatchNorm1d(
                num_features = out_channels
            ),
            nn.ReLU(),
            nn.Dropout(p = drop_p)
        ]
        return nn.Sequential(*Conv_Block)

    def forward(self, x):
        fx = self.Conv_Blocks(x)
        return self.ReLU(fx + x)

class Reduction(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop_p):
        super(Reduction, self).__init__()
        self.reduction_layer = nn.Sequential(
            nn.Conv1d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = 1,
                padding = 0
            ),
            nn.BatchNorm1d(
                num_features = out_channels
            ),
            nn.ReLU(),
            nn.Dropout(p = drop_p) )

    def forward(self, x):
        return self.reduction_layer(x)


class MyResNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, kernel_size, block_list, n_layers, drop_p):

        super(MyResNet, self).__init__()


        self.encoder, final_channel = self.make_encoder(in_channels, out_channels, kernel_size, block_list, n_layers, drop_p)

        self.adapt = nn.AdaptiveAvgPool1d(output_size = 1)

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features = final_channel,
                out_features = n_classes)
        )

    def make_encoder(self, in_channels, out_channels, kernel_size, block_list, n_layers, drop_p):
        encoder = nn.Sequential()
        
        for idx, blocks in enumerate(block_list):
            encoder.add_module(f'Reduction{idx+1}', Reduction(in_channels, out_channels, kernel_size, drop_p))
            for n in range(blocks):
                encoder.add_module(f'Resblock{idx+1}_{n+1}', Residual_Block(out_channels, out_channels, kernel_size, n_layers, drop_p))

            in_channels = out_channels
            out_channels *= 2
            
                
            
            
        return encoder, in_channels

    def forward(self, x):
        x = self.encoder(x)
        x = self.adapt(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.classifier(x)
        return x
        
        
