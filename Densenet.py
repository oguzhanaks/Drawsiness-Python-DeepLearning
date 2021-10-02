from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
import os
import numpy as np
from matplotlib import pyplot as plt
from imutils.video import VideoStream
import imutils
import time
import cv2
import dlib
from glob import glob
import math



if __name__ == '__main__': 
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    print("Device: ",device)
    data_transforms = {
            'trainface': transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                #normalizasyon
            ]),
            'testface': transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            'ktestface': transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
    }
    data_dir = 'D:\deneme1'   #veri yolu
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['trainface', 'testface']}
        
    train_loader = torch.utils.data.DataLoader(image_datasets['trainface'], batch_size=32,
                                                      shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(image_datasets['testface'], batch_size=32 ,
                                                      shuffle=False, num_workers=4)
    
    classes = ('alert','tired')
   
    
    
    def imshow(img):
        img = img / 2 + 0.5
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        plt.show()
        
    image_iter = iter(train_loader)
    images, _ = image_iter.next()
    imshow(torchvision.utils.make_grid(images[:4]))
  
    
    
    
    class Bottleneck(nn.Module):
  
        expansion = 4
        
        def __init__(self, in_channels, growth_rate):
            super(Bottleneck, self).__init__()
            zip_channels = self.expansion * growth_rate
            self.features = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, zip_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(zip_channels),
                nn.ReLU(True),
                nn.Conv2d(zip_channels, growth_rate, kernel_size=3, padding=1, bias=False)
            )
            
        def forward(self, x):
            out = self.features(x)
            out = torch.cat([out, x], 1)
            return out        
        
        
    class Transition(nn.Module):
   
        def __init__(self, in_channels, out_channels):
            super(Transition, self).__init__()
            self.features = nn.Sequential(
                nn.BatchNorm2d(in_channels  ),
                nn.ReLU(True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.AvgPool2d(2)
            )
            
        def forward(self, x):
            out = self.features(x)
            return out
        
        
    class DenseNet(nn.Module):
   
        def __init__(self, num_blocks, growth_rate=12, reduction=0.5, num_classes=2):
            super(DenseNet, self).__init__()
            self.growth_rate = growth_rate
            self.reduction = reduction
            
            num_channels = 2 * growth_rate
            
            self.features = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
            self.layer1, num_channels = self._make_dense_layer(num_channels, num_blocks[0])
            self.layer2, num_channels = self._make_dense_layer(num_channels, num_blocks[1])
            self.layer3, num_channels = self._make_dense_layer(num_channels, num_blocks[2])
            self.layer4, num_channels = self._make_dense_layer(num_channels, num_blocks[3], transition=False)
            self.avg_pool = nn.Sequential(
                nn.BatchNorm2d(num_channels),
                nn.ReLU(True),
                nn.AvgPool2d(4),
            )
            self.classifier = nn.Linear(num_channels, num_classes)
            
            self._initialize_weight()
            
        def _make_dense_layer(self, in_channels, nblock, transition=True):
            layers = []
            for i in range(nblock):
                layers += [Bottleneck(in_channels, self.growth_rate)]
                in_channels += self.growth_rate
            out_channels = in_channels
            if transition:
                out_channels = int(math.floor(in_channels * self.reduction))
                layers += [Transition(in_channels, out_channels)]
            return nn.Sequential(*layers), out_channels
        
        def _initialize_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
        
        def forward(self, x):
            out = self.features(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out
   

    def DenseNet1():
        return DenseNet([6,12,32,32], growth_rate=32)
  
    net = DenseNet1()
    net.cuda()
    # print(net)
    if device == 'cuda':
        net = nn.DataParallel(net)
       
        torch.backends.cudnn.benchmark = True
    x = torch.randn(1, 3, 32, 32).cuda()
    y = net(x)
    # print(y.shape)
    lr = 1e-1
    momentum = 0.9
    weight_decay = 1e-4
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
   
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225])
    total_step = len(train_loader)
    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            net.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 2 == 0:
                    print("epoch: {} {}/{}".format(epoch,batch_idx,total_step))
        print("Accuracy train %d %%"%(100*correct/total))   
        train_acc.append(100*correct/total)            
      
        return loss
        
    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
    
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print("Accuracy test %d %%"%(100*correct/total))
        test_acc.append(100*correct/total)
      
        return loss
    
    
    start_epoch = 0
    print('start_epoch: %s' % start_epoch)
   
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    for epoch in range(start_epoch,100):
        scheduler.step()
        train_loss = train(epoch)
        test_loss = test(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        start_epoch = epoch
        
    

    save_model = True
    fig, ax1 = plt.subplots()
    # plt.plot(loss_list,label = "Loss",color = "black")
    ax2 = ax1.twinx()
    ax1.plot(np.array(test_acc)/100,label = "Test Acc",color="red")
    # ax2.plot(np.array(train_acc)/100,label = "Train Acc",color= "red")
    
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('Epoch')
    fig.tight_layout()  
    plt.title("Loss vs Test Accuracy")
    plt.show()

    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0,
                    help="index of webcam on system")
    args = vars(ap.parse_args())
    
    
    print("-> Loading the predictor and detector...")
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    
    
    
    
    
    
    print("-> Starting Video Stream")
    vs = VideoStream(src=args["webcam"]).start()
    time.sleep(1.0)
    
    
    i = 0
    def rgb2gray(rgb):

        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray 
    detector = dlib.get_frontal_face_detector()
    while True:
        while True:
            frame = vs.read()
            gray = imutils.resize(frame, width=450)
            inx = 0
           
            if len(detector(gray)) ==True:
                cv2.imwrite(r'C:\Users\oguzh\Desktop\kendim2\ktestface\test\%d'%i+".jpg",gray)
                plt.imshow(gray)
                i = i +1
                if i == 32:
                    i=0
                    break
        
        pngs = glob(r'C:\Users\oguzh\Desktop\kendim2\ktestface\test*.jpg')
        
        for j in pngs:
            img = cv2.imread(j)
            
        
        
            faces = detector(img)  
            ind = 0
         
            tl_col = faces[ind].tl_corner().x
            tl_row = faces[ind].tl_corner().y
            br_col = faces[ind].br_corner().x
            br_row = faces[ind].br_corner().y
            tl_h = faces[ind].height()
            tl_w = faces[ind].width()
            
            x=img[tl_row:tl_row + tl_h, tl_col:tl_col + tl_w, ::-1]
       
      
        
            rs = cv2.resize(x , (32,32))
            gray = rgb2gray(rs)
            cv2.imwrite(j[:-3]+ 'jpg',gray)
            
        kdata_dir =r'C:\Users\oguzh\Desktop\kendim2' 
        
                      
     
        kimage_datasets = {x: datasets.ImageFolder(os.path.join(kdata_dir, x),
                                                  data_transforms[x])
                          for x in ['ktestface']}
        ktest_loader = torch.utils.data.DataLoader(kimage_datasets['ktestface'], batch_size=32 ,shuffle=False, num_workers=4)
        
       
        dataiter = iter(ktest_loader)
    
   
        
        
       
        
        outputs = net(images.cuda())
        _, predicted = torch.max(outputs.cpu(), 1)
        # print(classes[predicted[0]])
        x = 0
        y = 0
     
        for i in range(32):
            
                if classes[predicted[i]] == 'alert':
                    x = x+1
                else:
                    y = y+1
        if x>y:
            print("Uyanık ")
        else:
            print("Yorgun")
                    
        print( ''.join('%32s' % classes[predicted[j]]
                                            for j in range(32)))
        i = 0