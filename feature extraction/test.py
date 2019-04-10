import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms

num_classes = 10
#alexNet neuralnetwork class

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





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
    parser.add_argument('--image', required=True, help='Full path to image for inference')
    args = parser.parse_args()
    #setting the parameters for image location and model checkpoint path

    PATH =args.checkpoint
    impath=args.image
    #set device gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = AlexNet()
    #loading model
    print("loading model")
    print(net.classifier)
    net.load_state_dict(torch.load(PATH, map_location='cpu'))

    net.eval()  # Set model to evaluate mode
    print("model set to eval mode")



    input_size = 224

    data_transforms = {transforms.Compose({
        transforms.Resize(224),

        transforms.CenterCrop(224),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor()
    })}
    tn = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # preparing data for inference
    pat = "pic.jpeg"
    image = cv2.imread(impath)
    im = cv2.resize(image, (28, 28))# converts to 28 by 28 image like mnist
    i = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) #convert to hsv
    LOW = np.array([0, 0, 0])
    UP = np.array([255, 150, 150])
    mask = cv2.inRange(i, LOW, UP)#masking the image to identify the written number
    cv2.imwrite(pat, mask)

    image = Image.open(pat)
    im = tn(image)
    imm = im.unsqueeze(0) #adding 1 more channel
    #inp = Image.open(pat)

    #transform = transforms.Compose(data_transforms)
    outputs = net(imm)
    _, preds = torch.max(outputs, 1)
    print(preds)
    print(preds.item())


    classfic= preds.data[0][0]
    index=int(classfic)
    names = net.classes
    print(names[index])


