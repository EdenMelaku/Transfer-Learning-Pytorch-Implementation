import base64

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, make_response
from torchvision import transforms

app = Flask(__name__)

num_classes = 10


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


@app.route('/')
def index():
    print("got request ")
    a = request.args.get('image')

    # image_data = request.REQUEST["image_data"].decode("base64") # The data image
    print(a)
    print()
    c = a.strip()
    print(base64.urlsafe_b64decode(c))
    with open('number.jpeg', "wb") as f:
        f.write(base64.urlsafe_b64decode(a))

    response = make_response()
    response.content_type = 'text'


    fh = open("saved.jpeg", "wb")
    fh.write(base64.urlsafe_b64decode(a))
    fh.close()

    PATH = "/home/eden/newMODEL/modelAlexNet.pth"
    device = torch.device("cpu")

    net = AlexNet()
    print("loading model")
    print(net.classifier)
    net.load_state_dict(torch.load(PATH, map_location='cpu'))

    net.eval()  # Set model to evaluate mode
    print("model set to eval mode")


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
    pat = "saved.jpeg"
    image = cv2.imread(pat)
    im = cv2.resize(image, (28, 28))
    i = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    LOW = np.array([0, 0, 0])
    UP = np.array([255, 150, 150])
    mask = cv2.inRange(i, LOW, UP)
    cv2.imwrite(pat, mask)

    image = Image.open(pat)
    im = tn(image)
    imm = im.unsqueeze(0)
    inp = Image.open(pat)

    transform = transforms.Compose(data_transforms)

    outputs = net(imm)
    # outputs=net(process_image(pat))
    _, preds = torch.max(outputs, 1)
    print(preds.item())
    response = str(preds.item())

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')


