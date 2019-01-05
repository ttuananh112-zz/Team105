import torch as tr, cv2, numpy as np
device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

class CNN32(tr.nn.Module):
    def __init__(self):
        super(CNN32, self).__init__() #3x32x32
        self.pool = tr.nn.MaxPool2d(2, 2)

        self.conv1 = tr.nn.Conv2d(3, 16, 5) #28 (in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=0)
        #pool 14
        self.conv2 = tr.nn.Conv2d(16, 32,3) #12
        #pool 6
        self.conv3 = tr.nn.Conv2d(32, 64, 3) #4
        #pool 2
        self.fc1 = tr.nn.Linear(64 *2 *2, 128) #
        self.fc2 = tr.nn.Linear(128, 64)
        self.fc3 = tr.nn.Linear(64, 3)


    def forward(self, X): #3*32*32 #3*227*227
        X = tr.nn.functional.relu(self.conv1(X))
        X = self.pool(X)
        X = tr.nn.functional.relu(self.conv2(X))
        X = self.pool(X)
        X = tr.nn.functional.relu(self.conv3(X))
        X = self.pool(X)

        X = X.view(X.size(0), -1)

        X = tr.tanh(self.fc1(X))
        X = tr.tanh(self.fc2(X))
        X = tr.nn.functional.softmax(self.fc3(X), dim=1)
        return X

net= CNN32().to(device)
net.load_state_dict(tr.load('param/sign_classi_param32_small'))

def predict(img):
    new_size = 32
    img = cv2.resize(img, (new_size, new_size))
    if img.max()>1:
        img = np.array(img, dtype= np.float32) / 255.

    img= img.reshape(1,new_size,new_size,3).transpose((0,3,1,2))

    with tr.no_grad():
        img = tr.from_numpy(img).to(device)
        output= net(img)
        output= tr.argmax(output)

    return int(output) #0= not turn; 1= turn right, 2= turn left
