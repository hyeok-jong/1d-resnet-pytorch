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
        
        self.skip_connection = nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = padding
        )
        
        Conv_Blocks = list()
        for n in range(n_layers):
            if n != 0:
                in_channels = out_channels
            Conv_Blocks.append(
                self.BaseModule(in_channels, out_channels, kernel_size, padding, drop_p, last = n == n_layers - 1)
            )
        self.Conv_Blocks = nn.Sequential(*Conv_Blocks)



        self.ReLU = nn.ReLU()


    def BaseModule(self, in_channels, out_channels, kernel_size, padding, drop_p, last):
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
        if last:
            Conv_Block.pop(-2)
        return nn.Sequential(*Conv_Block)

    def forward(self, x):
        fx = self.Conv_Blocks(x)
        return self.ReLU(fx + self.skip_connection(x))

class Reduction(nn.Module):
    def __init__(self, channels, drop_p):
        super(Reduction, self).__init__()
        self.reduction_layer = nn.Sequential(
            nn.Conv1d(
                in_channels = channels,
                out_channels = channels,
                kernel_size = 3,
                stride = 2,
                padding = 0
            ),
            nn.BatchNorm1d(
                num_features = channels
            ),
            nn.ReLU(),
            nn.Dropout(p = drop_p) )

    def forward(self, x):
        return self.reduction_layer(x)


class MyResNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, kernel_size, block_list, n_layers, drop_p):
        '''
        in_channels : input shape (batch, in_channels, n_length)
        n_classes : output shape e.g) binary classification can 1 or 2
        block_list : list of blocks
        n_layers : number of layers in one block
        drop_p : proportion in dropout

        Features
        1. Batch Normalization included,
        2. All the 1D-CNN have same parameters,
        3. input shape should follow (batch, channel, length of sequence).
        4. No activation function on last layer, so when compute loss, one should use nn.CrossEntropyLoss or nn.BCEWithLogitsLoss
           which including activation function.
        5. 
        '''
        super(MyResNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.drop_p = drop_p

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
            
            for n in range(blocks):
                encoder.add_module(f'Resblock{idx+1}_{n+1}', Residual_Block(in_channels, out_channels, kernel_size, n_layers, drop_p))
                in_channels = out_channels
            
                
            encoder.add_module(f'Reduction{idx+1}', Reduction(out_channels,drop_p))
            out_channels *= 2
            
        return encoder, out_channels//2

    def forward(self, x):
        x = self.encoder(x)
        x = self.adapt(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.classifier(x)
        return x
        
        
