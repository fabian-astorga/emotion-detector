import torch.nn as nn
import torch.nn.functional as F

class ModelGenerator(nn.Module):
    def __init__(self):

        super(ModelGenerator,self).__init__()

        #convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3)

        #Max pooling layers
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)

        #batch normalization layers
        self.norm1 = nn.BatchNorm2d(num_features=10)
        self.norm2 = nn.BatchNorm2d(num_features=10)

        #full connected layers
        self.fc1 = nn.Linear(in_features=810,out_features=50)
        self.fc2 = nn.Linear(in_features=50,out_features=7)
        
        #Dropout layer
        self.dropout = nn.Dropout(0.3)

        #Softmax
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, input):
        # Neural network model generation
        out = self.conv1(input)
        out = F.relu(out)
<<<<<<< HEAD

=======
>>>>>>> 21813058457294dd6aac94de1af16c5b0d5c9354
        out = self.conv2(out)
        out = self.pool2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = F.relu(out)
<<<<<<< HEAD

=======
>>>>>>> 21813058457294dd6aac94de1af16c5b0d5c9354
        out = self.conv4(out)
        out = self.norm1(out)
        out = self.pool4(out)
        out = F.relu(out)

        out = self.dropout(out)
<<<<<<< HEAD
        
        out = out.view(out.size(0), -1)

=======
        out = out.view(-1, 810)
>>>>>>> 21813058457294dd6aac94de1af16c5b0d5c9354
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.log_softmax(out)

        return out