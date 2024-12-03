import torch
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
                nn.Tanh(),
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
                nn.Softmax(dim=1),
            )



            # fully connected layer, output 10 classes

            self.out = nn.Sequential(
                nn.Flatten(),
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
    print("CLASSIFIER 4:")
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

    print(len(training_data))
    print(len(validation_data))
    print(len(test_data))

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle = True)

    model = OF_ES().to(device)
    print(model)

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_model = None
    train_loss = []
    validation_acc = []
    best_acc=None
    epochs = 10000
    no_improvement = 5

    for t in range(epochs):
        print("\n")
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        epoch_loss = []
        test_loss = 0
        corr = 0
        size = len(test_dataloader)
        for X, y in test_dataloader:
            optimizer.zero_grad()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            corr += (pred.argmax(1) == y).type(torch.float).sum().item()
            optimizer.step()
            epoch_loss.append(test_loss)
        test_loss /= size
        corr /= size
        train_loss.append(torch.tensor(epoch_loss).mean())
        model.eval()
        valid_loader=torch.utils.data.DataLoader(validation_data,batch_size=len(validation_data),shuffle=False)
        X,y = next(iter(valid_loader))
        validation_acc.append(corr)
        print(f"Test Error: \n Accuracy: {(corr):>0.1f}%, Avg loss: {test_loss:>8f}")
        if best_acc is None or corr>best_acc:
            print(f"New best epoch: {t}, Accuracy: {(corr):>0.1f}%")
            best_acc = corr
            best_model = model.state_dict()
            best_epoch = t
        if best_epoch + no_improvement <= t:
            print("No improvement for ",no_improvement, "epochs")
            break

    print("Done!")

    torch.save(model.state_dict(), "Classifier_4.pth")
    print("Saved PyTorch Model State to Classifier_4.pth")
    #save the model

    model = OF_ES()
    model.load_state_dict(torch.load("Classifier_4.pth"))
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
    #img = input("Please enter a filepath: ")
    test_img = Image.open("img_1.jpg")
    #THROW EXCEPTION IF NOT THERE
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


if __name__ == "__main__":
    main()
