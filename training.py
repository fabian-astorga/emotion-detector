import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from plain_dataset import PlainDataset
from model_generator import ModelGenerator
from downloader import Downloader

#Training parameters
epochs = 50
learning_rate = 1e-3
batch_size = 1500
weight_decay = 1e-4

#File paths
traincsv_file = './data/train.csv'
validationcsv_file = './data/val.csv'
train_img_dir = './data/train/'
validation_img_dir = './data/val/'

# Check if GPU is available
device = ""
if (torch.cuda.is_available()):
    device = "cuda:0"
else:
    device = "cpu"
torch.device(device)

class Training():
    def __init__(self, training_loader, validation_loader, criterion, optimizer):
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.criterion = criterion
        self.optimizer = optimizer

    def train_cnn(self):
        print("Starting Training...")
        for e in range(epochs):
            train_loss = 0
            validation_loss = 0
            train_correct = 0
            val_correct = 0
            # Train the model
            net.train()
            for data, labels in self.training_loader:
                data, labels = data.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = net(data)
                loss = self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                _, preds = torch.max(outputs,1)
                train_correct += torch.sum(preds == labels.data)

            #validate the model#
            net.eval()
            for data,labels in self.validation_loader:
                data, labels = data.to(device), labels.to(device)
                val_outputs = net(data)
                val_loss = self.criterion(val_outputs, labels)
                validation_loss += val_loss.item()
                _, val_preds = torch.max(val_outputs,1)
                val_correct += torch.sum(val_preds == labels.data)

            train_loss = train_loss/len(train_dataset)
            train_acc = train_correct.double() / len(train_dataset)
            validation_loss =  validation_loss / len(validation_dataset)
            val_acc = val_correct.double() / len(validation_dataset)

            print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}%'
                                                            .format(e+1, train_loss,validation_loss,train_acc*100, val_acc*100))

        torch.save(net.state_dict(),'./models/new_model.pt'.format(epochs,batch_size,lr))
        print("Training finished")

if __name__ == '__main__':
  
    # Generates images and val.csv
    # downloader = Downloader('./data')
    # downloader.save_images_from_csv('train')
    # downloader.save_images_from_csv('val')

    net = ModelGenerator() #Creating the model
    net.to(device) # Loading it to gpu/cpu

    print("Device: ", device)
    print("Model architecture: ", net)

    transformation = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)) ] )
    train_dataset = PlainDataset(csv_file=traincsv_file, img_dir=train_img_dir, datatype='train', transform=transformation)
    validation_dataset = PlainDataset(csv_file=validationcsv_file, img_dir=validation_img_dir, datatype='val', transform=transformation)
    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader =  DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    training = Training(training_loader, validation_loader, criterion, optimizer)
    training.train_cnn() #Training the model