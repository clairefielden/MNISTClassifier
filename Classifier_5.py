import torch
import os
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.models as models
import matplotlib.pyplot as plt

import tarfile
import gzip
import shutil
import json

import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

best_accuracy = 0

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def gunzip(file_path,output_path):
    with gzip.open(file_path,"r") as f_in, open(output_path,"wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    global best_accuracy
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    if(correct>best_accuracy):
        correct = best_accuracy
        torch.save(model.state_dict(), "Classifier_5.pth")
        print("Saved new best learner to Classifier_5.pth")
        # save the model
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class MNIST_Dataset(torch.utils.data.Dataset):
  # 784 tab-delim pixel values (0-255) then label (0-9)
  def __init__(self, src_file):
    all_xy = np.loadtxt(src_file, usecols=range(785),
      delimiter="\t", comments="#", dtype=np.float32)

    tmp_x = all_xy[:, 0:784]  # all rows, cols [0,783]
    tmp_x /= 255
    tmp_x = tmp_x.reshape(-1, 1, 28, 28)
    tmp_y = all_xy[:, 784]

    self.x_data = torch.tensor(tmp_x, dtype=torch.float32).to(device)
    self.y_data = torch.tensor(tmp_y, dtype=torch.int64).to(device)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    lbl = self.y_data[idx]  # no use labels
    pixels = self.x_data[idx]
    return (pixels, lbl)

# Define model
class OF_ES(nn.Module):
    def __init__(self):
            super(OF_ES, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=4,
                    kernel_size=5,
                    stride=1,
                    padding=1
                ),
                #input layer
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            )

            #average pooling layer comes second - subsampling
            #second layer


            #3rd layer
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                in_channels=4,
                out_channels=12,
                kernel_size=5,
                stride=1,
                padding=1
                ),
                nn.LeakyReLU(),
                nn.BatchNorm2d(12),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            )
            # local averaging
            #2-to-1 subsampling
            #fourth layer

            #adding fifth layer that will distribute into more feature maps
            #apply batchnorm between the 2 hidden layers

            self.conv3 = nn.Sequential(
                nn.Conv2d(
                    in_channels=12,
                    out_channels=8,
                    kernel_size=4,
                    stride=2,
                    padding=2
                ),
                nn.LeakyReLU(),
            )

            # fully connected layer, output 10 classes

            self.out = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.1),
                # Dropout with probability of 0.1 to avoid overfitting - OBD
                nn.Linear(72, 10),
            )

            #need to flatten
            #using linear = weights affected, everything connected

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten the output of conv2 to (batch_size, 12 * 4 * 4)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        # return output, x    # return x for visualization
        return output

def convert(img_file, label_file, txt_file, n_images):
  lbl_f = open(label_file, "rb")   # MNIST labels (digits)
  img_f = open(img_file, "rb")     # MNIST pixel values
  txt_f = open(txt_file, "w")      # output file to write to

  #Discarding binary pixel and label files headers
  img_f.read(16)   # discard header info
  lbl_f.read(8)    # discard header info

  #Reading binary files
  for i in range(n_images):   # number images requested
    lbl = ord(lbl_f.read(1))  # get label (unicode, one byte)
    #writing to text file
    for j in range(784):  # get 784 vals from the image file
      val = ord(img_f.read(1))
      txt_f.write(str(val) + "\t")
      #write the pixel values (tab delim) then label val
    txt_f.write(str(lbl) + "\n")
  img_f.close(); txt_f.close(); lbl_f.close()

def main():
    print("CLASSIFIER 5:")
    printed = input("Has the data been extracted already?(Y/N)")
    if(printed!="Y"):
            # extract .gz MNIST data from MNIST/raw file
            gunzip('MNIST/raw/train-images-idx3-ubyte.gz', 'unzippedBinary/train-images-idx3-ubyte.bin')
            gunzip('MNIST/raw/train-labels-idx1-ubyte.gz', 'unzippedBinary/train-labels-idx1-ubyte.bin')
            gunzip('MNIST/raw/t10k-labels-idx1-ubyte.gz', 'unzippedBinary/t10k-labels-idx1-ubyte.bin')
            gunzip('MNIST/raw/t10k-images-idx3-ubyte.gz', 'unzippedBinary/t10k-images-idx3-ubyte.bin')

            convert('unzippedBinary/train-images-idx3-ubyte.bin', 'unzippedBinary/train-labels-idx1-ubyte.bin',
                'unzippedBinary/trainingData.txt', 60000)
            convert('unzippedBinary/t10k-images-idx3-ubyte.bin', 'unzippedBinary/t10k-labels-idx1-ubyte.bin',
                'unzippedBinary/validationData.txt', 10000)

    # Download test data from open datasets.
    training_data = MNIST_Dataset("unzippedBinary/trainingData.txt")
    test_data = MNIST_Dataset("unzippedBinary/validationData.txt")

    len_train = int(0.6*len(training_data))
    len_valid = int(len(training_data)-len_train)
    #split up the training and validation sets
    training_data, validation_data = torch.utils.data.random_split(training_data, (len_train, len_valid))


    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle = True)

    model = OF_ES().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum = 0.8)


    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    print("Done!")

    torch.save(model.state_dict(), "Classifier_5.pth")
    print("Saved PyTorch Model State to Classifier_5.pth")
    #save the model

    model = OF_ES()
    model.load_state_dict(torch.load("Classifier_5.pth"))
    model.eval()
    #retrieve the model

    classes = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]

    #obtain input image
    img = input("Please enter a filepath: ")


    test_img = Image.open(img)

    if (os.path.exists(img)):
        test_img = Image.open(img)
        test_img_data = np.asarray(test_img)
        plt.imshow(test_img_data)
        plt.show()

        # model expects 28x28 greyscale image
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()
        ])

        # standard ImageNet normalization
        transform_normalize = transforms.Normalize((0.5,),(0.5,))

        transformed_img = transform(test_img)
        input_img = transform_normalize(transformed_img)
        input_img = input_img.unsqueeze(0) # the model requires a dummy batch dimension

        with torch.no_grad():
            pred = model(input_img)
            predicted = classes[pred[0].argmax(0)]
            print(f'Classifier: {predicted}')
    else:
        print("File does not exist.")
        exit(1)

if __name__ == "__main__":
    main()
