import argparse
import cv2

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from torchvision import datasets
from torchvision.transforms import transforms

num_classes=10



class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    size = 256, 256
    crop_size = 224

    im = Image.open(image_path)

    im.thumbnail(size)

    left = (size[0] - crop_size) / 2
    top = (size[1] - crop_size) / 2
    right = (left + crop_size)
    bottom = (top + crop_size)

    im = im.crop((left, top, right, bottom))

    np_image = np.array(im)
    np_image = np_image / 255

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    np_image =(np.array(np_image) - means) /stds

    pytorch_np_image = np_image.transpose(2, 0, 1)


    return pytorch_np_image


if __name__=="__main__":

    PATH="/home/eden/newMODEL/modelAlexNet.pth"
    device = torch.device("cpu")

    net=AlexNet()
    print("loading model")
    print(net.classifier)
    net.load_state_dict(torch.load(PATH, map_location='cpu'))

    net.eval()  # Set model to evaluate mode
    print("model set to eval mode")

    #model = AlexNet()
    #optimizer= torch.optim.SGD (model.parameters(), lr=0.001, momentum=0.9)

    #checkpoint = torch.load(PATH)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
   # loss = checkpoint['loss']

    #model.eval()



    input_size=224

    data_transforms = {transforms.Compose({
        transforms.Resize(224),

        transforms.CenterCrop(224),



        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        transforms.ToTensor()
    })}
    tn = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
transforms.CenterCrop(224),        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # Create training and validation datasets
    pat = "num.jpg"
    image = cv2.imread(pat)
    im = cv2.resize(image, (28, 28))
    i = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    LOW = np.array([0, 0, 0])
    UP = np.array([255, 150, 150])
    mask = cv2.inRange(i, LOW, UP)
    cv2.imwrite(pat,mask)

    image= Image.open(pat)
    im=tn(image)
    imm=im.unsqueeze(0)
    inp=Image.open(pat)
   # inputs = input.to(device)
    transform=transforms.Compose(data_transforms)
#    inn=transform(inp)
    outputs=net(imm)
    #outputs=net(process_image(pat))
    _, preds = torch.max(outputs, 1)
    print(preds)
    print(preds.item())


    sm = torch.nn.Softmax()
    probablities = sm(outputs)
    n=probablities.detach().numpy()
    result=[x *100 for x in n ]
    print(probablities.detach().numpy())

    m=n*100
    from itertools import chain
    a=list(chain.from_iterable(m))
    print("\n")

    i = 0
    for x in a:
        print("probablity of  " + str(i) + "====" + str(x))
        i += 1
