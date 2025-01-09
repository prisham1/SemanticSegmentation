#PyTorch library for building neural network 
import torch 
# Contains modules for creating neural networks
import torch.nn as nn 
#Contains functions for image transformations
import torchvision.transforms.functional as TF

#Class defines a double convolutional layer 
class DoubleConv(nn.Module): 
    #Method initializes two convolutional layers with batch normalization and ReLU activation
    def __init__(self, in_channels, out_channels):
        # Call the parent class (nn.Module) constructor
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            #Same Convutional (input height and width is the same after the convolution)
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True), 
            #Maintains the same number of channels for the input and the output
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )

    #Defines the forward pass through two convolutional layers
    def forward(self, x):
        return self.conv(x) 
    

class UNET(nn.Module):
    #Initializes the layers and components of U-NET
    #Feature represents the number of output channels for each convolutional layer at different levels of the network 
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512],): 
        super(UNET, self).__init__()
        # Store in ModuleList to be able to perofrm modle.eval, etc.
        #List stores upsampling layers
        self.ups = nn.ModuleList()
        #List stores downsampling layers
        self.downs = nn.ModuleList()
        #Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET adds DoubleConv layers to the downsampling path
        for feature in features: 
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature 
        
        #Up part of UNET add ConvTranspose and DoubleConv layers for the upsampling path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            #Upsampling followed by two convolutional layers
            self.ups.append(DoubleConv(feature * 2, feature))  


        #Defines bottleneck layer between downsampling and upsampling 
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        #Does not change the height or width of an image, only changes the number of channels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self,x):
        #Stores and uses skip connections to combine low-level and high-level features
        skip_connections = []

        for down in self.downs: 
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for index in range(0, len(self.ups), 2): 
            #ConvTranspose2d
            x = self.ups[index](x)
            skip_connection = skip_connections[index//2]

            if x.shape != skip_connection.shape: 
                #Skips the batch_size and the number of channels, only takes the height and width
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection,x), dim=1)
            #Only works for inputs that are divisible by 16
            x = self.ups[index+1](concat_skip)

        return self.final_conv(x)

def test(): 
    x = torch.randn((3,1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    #Ensures the output shape matches the input shape
    assert preds.shape == x.shape 

if __name__ == "__main__":
    test()



