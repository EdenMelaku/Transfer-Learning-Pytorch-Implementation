import base64
import io

import cv2

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from cv2.cv2 import UMat
from flask import Flask, request, make_response
from torchvision import transforms

app = Flask(__name__)


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

@app.route('/')
def index():
    print("got request ")
    a = request.args.get('image')

    #image_data = request.REQUEST["image_data"].decode("base64") # The data image
    print(a)
    #imgdata = base64.b64decode(a)
    #b64_string = a.encode().decode()
    #image = np.fromstring(a, np.uint8)
    # reconstruct image as an numpy array

    #cv2.imshow('image ',image)


#    image = Image.open(io.BytesIO(a))
    #image = cv2.imread(image)
    response = make_response()
    response.content_type = 'text'
#    byte=bytearray(a)
   # data_mat(a, true)
   #mat = np.asarray(a, dtype=np.uint8)
   #image = np.fromarray(mat, np.uint8)

    #image=cv2.imdecode(a,1)

    by=bytearray(a,'utf-8')
    by.extend(map(ord,a ))
    imgd=base64.b64decode(a.encode('base64'))
    imgdata =base64.b64encode(by)
#    imaged=a.decode('base64')
    #nparr = np.fromstring(a.encode('base64').decode('base64'), np.uint8)
    #img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    #cv2.imwrite("/home/eden/Pictures/mod/imag.jpeg", img)

    fh = open("/home/eden/Pictures/mod/Saved.jpeg", "wb")
    fh.write(imgdata)
    fh.close()


    PATH = "/home/eden/newMODEL/modelAlexNet.pth"
    device = torch.device("cpu")

    net = AlexNet()
    print("loading model")
    print(net.classifier)
    net.load_state_dict(torch.load(PATH, map_location='cpu'))

    net.eval()  # Set model to evaluate mode
    print("model set to eval mode")

    # model = AlexNet()
    # optimizer= torch.optim.SGD (model.parameters(), lr=0.001, momentum=0.9)

    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    # model.eval()

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
    # Create training and validation datasets
    pat = "/home/eden/Pictures/mod/imageToSave.jpeg"
    image = cv2.imread(pat)
    im = cv2.resize(image, (28, 28))
    i = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    LOW = np.array([0, 0, 0])
    UP = np.array([255, 150, 150])
    mask = cv2.inRange(i, LOW, UP)
    cv2.imwrite(pat, mask)

    image = Image.open(pat)
    im = tn(image)
    imm = im.unsqueeze(0)


    outputs = net(imm)

    _, preds = torch.max(outputs, 1)
    print(preds)
    response=preds.item()
    return response


if __name__ == '__main__':

   app.run(host='0.0.0.0')

