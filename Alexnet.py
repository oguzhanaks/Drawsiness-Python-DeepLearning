import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
from torchvision import transforms
import torch.utils.data
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import time
import cv2
import dlib
from glob import glob
import math
import os
import numpy as np
from matplotlib import pyplot as plt
from imutils.video import VideoStream
import imutils
import argparse
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

data_transforms=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])


train_data =datasets.ImageFolder(r'D:\deneme1\trainface',transform=data_transforms)
test_data=datasets.ImageFolder(r'D:\deneme1\testface',transform=data_transforms)


        
train_loader = torch.utils.data.DataLoader(train_data, batch_size=256,
                                                      shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=256 ,
                                                      shuffle=False, num_workers=2)

VALID_RATIO = 0.9

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(train_data, 
                                           [n_train_examples, n_valid_examples])
    
classes = ('alert','tired')







BATCH_SIZE = 256

train_iterator = data.DataLoader(train_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)



test_iterator = data.DataLoader(test_data, 
                                batch_size = BATCH_SIZE)


class AlexNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), 
            nn.MaxPool2d(2), 
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 192, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True),
            nn.Conv2d(192, 384, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h
    
    

OUTPUT_DIM = 10

model = AlexNet(OUTPUT_DIM)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)
        
        
model.apply(initialize_parameters)


class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device
        
        torch.save(model.state_dict(), 'init_params.pt')

    def range_test(self, iterator, end_lr = 10, num_iter = 100, 
                   smooth_f = 0.05, diverge_th = 5):
        
        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        
        iterator = IteratorWrapper(iterator)
        
        for iteration in range(num_iter):

            loss = self._train_batch(iterator)

            lrs.append(lr_scheduler.get_last_lr()[0])

            #update lr
            lr_scheduler.step()
            
            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
                
            if loss < best_loss:
                best_loss = loss

            losses.append(loss)
            
            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break
                       
        #reset model to initial parameters
        model.load_state_dict(torch.load('init_params.pt'))

                    
        return lrs, losses

    def _train_batch(self, iterator):
        
        self.model.train()
        
        self.optimizer.zero_grad()
        
        x, y = iterator.get_batch()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        y_pred, _ = self.model(x)
                
        loss = self.criterion(y_pred, y)
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels

    def get_batch(self):
        return next(self)
    
    

START_LR = 1e-7

optimizer = optim.Adam(model.parameters(), lr = START_LR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)



FOUND_LR = 1e-3

optimizer = optim.Adam(model.parameters(), lr = FOUND_LR)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
test_acc = []

def train(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred, _ = model(x)
        
        loss = criterion(y_pred, y)
        
        acc = calculate_accuracy(y_pred, y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


EPOCHS = 100

best_test_loss = float('inf')

for epoch in range(EPOCHS):
    
    start_time = time.monotonic()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
        
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

    
fig, ax1 = plt.subplots()
# plt.plot(loss_list,label = "Loss",color = "black")
ax2 = ax1.twinx()
ax1.plot(np.array(test_acc)/100,label = "Test Acc",color="green")
# ax2.plot(np.array(train_acc)/100,label = "Train Acc",color= "red")
ax1.legend()
ax2.legend()
ax1.set_xlabel('Epoch')
fig.tight_layout()
plt.title("Test Accuracy")
plt.show()
# ap = argparse.ArgumentParser()
# ap.add_argument("-w", "--webcam", type=int, default=0,
#                     help="index of webcam on system")
# args = vars(ap.parse_args())
    
    
# print("-> Loading the predictor and detector...")
# detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    
    
    
    
    
    
# print("-> Starting Video Stream")
# vs = VideoStream(src=args["webcam"]).start()
# time.sleep(1.0)
    
    
# i = 0
# def rgb2gray(rgb):

#     r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

#     return gray 
# detector = dlib.get_frontal_face_detector()
# while True:
#     while True:
#             frame = vs.read()
#             gray = imutils.resize(frame, width=450)
#             inx = 0
           
#             if len(detector(gray)) ==True:
#                 cv2.imwrite(r'C:\Users\oguzh\Desktop\kendim2\%d'%i+".png",gray)
#                 plt.imshow(gray)
#                 i = i +1
#                 if i == 256:
#                     i=0
#                     break
        
#     pngs = glob(r'C:\Users\oguzh\Desktop\kendim2*.png')
        
#     for j in pngs:
#         img = cv2.imread(j)
            
        
        
#         faces = detector(img)  
#         ind = 0
         
#         tl_col = faces[ind].tl_corner().x
#         tl_row = faces[ind].tl_corner().y
#         br_col = faces[ind].br_corner().x
#         br_row = faces[ind].br_corner().y
#         tl_h = faces[ind].height()
#         tl_w = faces[ind].width()
            
#         x=img[tl_row:tl_row + tl_h, tl_col:tl_col + tl_w, ::-1]
       
      
        
#         rs = cv2.resize(x , (32,32))
#         gray = rgb2gray(rs)
#         cv2.imwrite(j[:-3]+ 'png',gray)
             
#     kdata_dir =r'C:\Users\oguzh\Desktop\kendim2' 
        
                      
#     kimage_datasets=datasets.ImageFolder(r'C:\Users\oguzh\Desktop\kendim2',transform=data_transforms) 
#     # kimage_datasets = {x: datasets.ImageFolder(os.path.join(kdata_dir, x),
#     #                                               data_transforms[x])
#     #                       for x in ['ktestface']}
    
#     ktest_loader = torch.utils.data.DataLoader(kimage_datasets, batch_size=256 ,shuffle=False, num_workers=2)
        
       
#     dataiter = iter(ktest_loader)
    
   
        
        
       
        
#     outputs = model(x.cuda())
#     _, predicted = torch.max(outputs.cpu(), 1)
#     # print(classes[predicted[0]])
#     x = 0
#     y = 0
     
#     for i in range(256):
            
#             if classes[predicted[i]] == 'alert':
#                 x = x+1
#             else:
#                 y = y+1
#     if x>y:
#         print("Uyanık ")
#     else:
#         print("Yorgun")
                    
#     print( ''.join('%256s' % classes[predicted[j]]
#                                             for j in range(256)))
#     i = 0
    