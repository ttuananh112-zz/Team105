import torch as tr, cv2, numpy as np
device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")

class CNN32(tr.nn.Module):
    def __init__(self):
        super(CNN32, self).__init__() #3x32x32
        self.pool = tr.nn.MaxPool2d(2, 2)

        self.conv1 = tr.nn.Conv2d(3, 32, 5) #28 (in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=0)
        #pool 14
        self.conv2 = tr.nn.Conv2d(32, 64,3) #12
        #pool 6
        self.conv3 = tr.nn.Conv2d(64, 128, 3) #4
        #pool 2
        self.fc1 = tr.nn.Linear(128 *2 *2, 1024) #
        self.fc2 = tr.nn.Linear(1024, 512)
        self.fc3 = tr.nn.Linear(512, 3)


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
net.load_state_dict(tr.load('sign_classi_param_9932_half', map_location=device))

def predict(img, new_size=32):
    img = np.array(img, dtype= np.float32) / 255.
    img = cv2.resize(img, (new_size, new_size))

    img= img.reshape(1,new_size,new_size,3).transpose((0,3,1,2))

    with tr.no_grad():
        img = tr.from_numpy(img).to(device)
        output= net(img)
        output= tr.argmax(output)

    return int(output) #0= not turn; 1= turn right, 2= turn left

# with tr.no_grad():
#     # while True:
#     #     dir= raw_input("file directory: ")
#     #     if dir == '': break
#     for i in range(24,28):
#         dir= 'other imgs/o27.png'#'other imgs/o' + str(i) + '.png'
#
#         img= cv2.imread(dir)
#         img= np.flip(img,1)
#         cv2.imshow(str(predict(img)), cv2.resize(img, (150,150)))
#         cv2.waitKey(0)
