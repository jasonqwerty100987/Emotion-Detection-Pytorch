import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from PIL import Image
import io
import socket
import struct
import numpy
def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ELU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.input = conv_block(in_channels, 64)

        self.conv1 = conv_block(64, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = conv_block(64, 64, pool=True)
        self.res2 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = conv_block(64, 64, pool=True)
        self.res3 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop3 = nn.Dropout(0.5)

        self.classifier = nn.Sequential(
            nn.MaxPool2d(6), nn.Flatten(), nn.Linear(64, num_classes)
        )

    def forward(self, xb):
        out = self.input(xb)

        out = self.conv1(out)
        out = self.res1(out) + out
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.drop3(out)

        return self.classifier(out)

face_classifier = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
model_state = torch.load("./models/emotion_detection_model_state.pth")
class_labels = ["Angry", "Happy", "Neutral", "Sad", "Suprise"]
model = ResNet(1, len(class_labels))
model.load_state_dict(model_state)

hostname = socket.gethostname()
local_ip = "192.168.1.200"
server_socket = socket.socket()
port = 8000
server_socket.bind((local_ip, port))
server_socket.listen(0)
print("Waiting for connection from client, ip is {}".format(local_ip))
connection = server_socket.accept()[0]
print("Connected")
connectionf = connection.makefile('rb')
try:
    while True:
        image_len = struct.unpack('<L', connectionf.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        image_stream = io.BytesIO()
        image_stream.write(connectionf.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream)
        print("Client's img received")
        image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(image, 1.3, 5)
        results = []
        for (x, y, w, h) in faces:
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = image[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = tt.functional.to_pil_image(roi_gray)
                roi = tt.functional.to_grayscale(roi)
                roi = tt.ToTensor()(roi).unsqueeze(0)

                # make a prediction on the ROI
                tensor = model(roi)
                pred = torch.max(tensor, dim=1)[1].tolist()
                label = class_labels[pred[0]]
                results.append((label, [x, y, w, h]))
        if results:
            if len(results) > 5:
                results = results[:5]
            msg = " ".join([str(label)+","+str(coordinate[0])+","+str(coordinate[1])+","+str(coordinate[2])+","+str(coordinate[3]) for label, coordinate in results])
            connection.sendall(msg.encode())
        else:
            msg = "N/A"
            connection.sendall(msg.encode())
        print("Responded")
finally:
    connectionf.close()
    connection.close()
    server_socket.close()
