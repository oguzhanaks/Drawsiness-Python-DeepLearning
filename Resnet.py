import torch 
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os 
import torch.utils.data
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import dlib
from glob import glob

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: ",device)
    
 
    def read_images(path, num_img):
        array = np.zeros([num_img, 64*32])          #64 satır 32 sütundan oluşan bir resim (bunu bir depo olarak düşünebiliriz) 
        i = 0
        for img in os.listdir(path):
            img_path = path + "\\" + img
            img = Image.open(img_path, mode = "r")  # r ile okuma ya da yazma yapıyoruz
            
            data = np.asarray(img, dtype = "uint8")
            data = data.flatten()                   # #düzleştirme işlemi
            array[i,:] = data
            i += 1
        return array
    
    
            
    # train-alert verilerinin okunması
    train_alert_path = r"C:\Users\oguzh\Desktop\sondata32_64jpg\sondatatrainjpg\alert"
    num_train_alert_img = 5088
    train_alert_array = read_images(train_alert_path,num_train_alert_img)
    x_train_alert_tensor = torch.from_numpy(train_alert_array)                  # pytorch array ye çevirme
    print("x_train_alert_tensor: ",x_train_alert_tensor.size())
    y_train_alert_tensor = torch.zeros(num_train_alert_img,dtype = torch.long)
    print("y_train_alert_tensor: ",y_train_alert_tensor.size())
    
    
    
    # train-tired verilierinin okunması
    train_tired_path = r"C:\Users\oguzh\Desktop\sondata32_64jpg\sondatatrainjpg\tired"
    num_train_tired_img = 3120
    train_tired_array = read_images(train_tired_path,num_train_tired_img)
    x_train_tired_tensor = torch.from_numpy(train_tired_array)
    print("x_train_tired_tensor: ",x_train_tired_tensor.size())
    y_train_tired_tensor = torch.ones(num_train_tired_img,dtype = torch.long)
    print("y_train_positive_tensor: ",y_train_tired_tensor.size())
    
    
    
    # train verilerinin concat edilmesi
    x_train = torch.cat((x_train_alert_tensor, x_train_tired_tensor), 0)        
    y_train = torch.cat((y_train_alert_tensor, y_train_tired_tensor), 0)
    print("x_train: ",x_train.size())
    print("y_train: ",y_train.size())
    
    
    
   # test-alert verilerinin okunması 
    test_alert_path = r"C:\Users\oguzh\Desktop\sondata32_64jpg\sondatatestjpg\alert"
    num_test_alert_img = 624
    test_alert_array = read_images(test_alert_path,num_test_alert_img)
    x_test_alert_tensor = torch.from_numpy(test_alert_array)
    print("x_test_alert_tensor: ",x_test_alert_tensor.size())
    y_test_alert_tensor = torch.zeros(num_test_alert_img,dtype = torch.long)
    print("y_test_alert_tensor: ",y_test_alert_tensor.size())
    
    
    
    # test-tired verilierinin okunması
    test_tired_path = r"C:\Users\oguzh\Desktop\sondata32_64jpg\sondatatestjpg\tired"
    num_test_tired_img = 336
    test_tired_array = read_images(test_tired_path,num_test_tired_img)
    x_test_tired_tensor = torch.from_numpy(test_tired_array)
    print("x_test_tired_tensor: ",x_test_tired_tensor.size())
    y_test_tired_tensor = torch.ones(num_test_tired_img,dtype = torch.long)
    print("y_test_tired_tensor: ",y_test_tired_tensor.size())
    
    
    
    # test verilerinin concat edilmesi
    x_test = torch.cat((x_test_alert_tensor, x_test_tired_tensor), 0)
    y_test = torch.cat((y_test_alert_tensor, y_test_tired_tensor), 0)
    print("x_test: ",x_test.size())
    print("y_test: ",y_test.size())
    
  
    #%% fotograf kontrolü
    plt.imshow(x_train[15,:].reshape(64,32), cmap='gray')   # fotoğrefi denemek için batırma (1,2048 lik halden 64,32 ye getirdik.)
    
    # %% 
    
    num_classes = 2             # alert ve tired diye 2 sınıfımız var
 
    classes = ('alert','tired')
    # Hyper parametreler
    
    num_epochs = 2         # döngü sayım
    batch_size = 16
    learning_rate = 0.001
    
    train = torch.utils.data.TensorDataset(x_train,y_train)
    trainloader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
    
    test = torch.utils.data.TensorDataset(x_test,y_test)
    testloader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
    
    
    
    
   
    
    def conv3x3(in_planes, out_planes, stride = 1):     # conv3x3= 3x3 lük kernel olacak yani kernel_size = 3 olacak , layer deki nöron sayısı , imput img deki chanellerin sayısı , stride = pikseller üzrinde tarama yaparken seçilecek filtre sayısı
        return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
            
    def conv1x1(in_planes, out_planes, stride = 1):
        return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)
        
    class BasicBlock(nn.Module):
        
        expansion = 1
        
        def __init__(self,inplanes, planes, stride = 1, downsample = None):
            super(BasicBlock,self).__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)               #her layer de normalizasyon yapmak 
            self.relu = torch.nn.SiLU(inplace=False)      #inplace = True relu aktivasyonunu çağırınca sonucu yine kendisine eşitle 
            self.drop = nn.Dropout(0.05)                     # zayıf bilgileri seyretm yapıyorum.
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride
            
        def forward(self, x):                               #Basic Block ları forwad metodu ile birbirne bağlıyor
            identity = x
            
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.drop(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.drop(out)
            
            if self.downsample is not None:             # sizeler eşit olmadan toplanamayacağı için downsample yapıyoruz
                identity = self.downsample(x)
                
            out += identity                             # downsample yapılmış x ile out u sum 
            out = self.relu(out)
            return out
    class ResNet(nn.Module):
        
        def __init__(self, block, layers, num_classes = num_classes):
            super(ResNet,self).__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride = 2, padding = 3, bias= False) # stride=2  downsample yapılabilir artık
            self.bn1 = nn.BatchNorm2d(64)                                                      #64 imput
            self.relu = nn.ReLU(inplace = True)                         # inplace = True relu aktivasyonunu çağırınca sonucu kendisine eşitle 
            self.maxpool = nn.MaxPool2d(kernel_size= 3, stride = 2, padding = 1)
            self.layer1 = self._make_layer(block, 64, layers[0], stride = 1)
            self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))              # output 1,1 .. filtrenin boyutunu otamatik kendisi belitliyor
            self.fc = nn.Linear(256*block.expansion, num_classes)
            
            for m in self.modules():              # m nin içerisine conv1,bn1 vb. gibi değerler geliyor.
                if isinstance(m,nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu") # kaiming_normal_ sıfıra yakın sayıları init ediyor çok büyük yada çok küçük sayıları init etmiyor, m nin weight ini güncelliyoruz  
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight,1)           # tüm weight leri 1 e eşitle
                    nn.init.constant_(m.bias,0)             # bias false olduğundan 0 a eşitliyorum    
                    
        def _make_layer(self, block, planes, blocks, stride = 1):  # _make_layer= BasicBlock ları birbirine bağlıyor,blocks= kaç tane BasicBlocs inşa edeceğimi belirliyor.
            downsample = None
            if stride != 1 or self.inplanes != planes*block.expansion:
                downsample = nn.Sequential(
                        conv1x1(self.inplanes, planes*block.expansion, stride),
                        nn.BatchNorm2d(planes*block.expansion))
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes*block.expansion
            for _ in range(1,blocks):   # başlangıçta defauld olarak bir tane blok eklediğim için döngüyü bir eksik alıyorum
                layers.append(block(self.inplanes, planes))
            
            return nn.Sequential(*layers)
            
        
        def forward(self,x):
            
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = x.view(x.size(0),-1)            # kendisi hesaplayıp dolduracak
            x = self.fc(x)
            
            return x
            
   # model = ResNet(BasicBlock, [2,2,2])                # 2 tane basicblock kullanacağımız için 2,2,2
    
    model = ResNet(BasicBlock, [2,2,2]).to(device)      # gpu kullanırken 

    
   
    
    
    #%% 
                                                                        # optimizer : minimum noktayı bulma
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate) # Adam algoritması momentum değişiklerini saklar
    
    
    #%% train
    
    loss_list = []
    train_acc = []
    test_acc = []
    use_gpu = True
    
    total_step = len(trainloader)
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            
            images = images.view(batch_size,1,64,32)                            # 1 = resmin hiç rengi olmadığı anlamına geliyor
            images = images.float()
            
            # gpu
            if use_gpu:
                if torch.cuda.is_available():
                    model.cuda()
                    images, labels = images.to(device), labels.to(device)
                
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            # backward and optimization
            optimizer.zero_grad()                                       # her adımda sıfırlıyoruz
            loss.backward()
            optimizer.step()
            
            if i % 2 == 0:
                print("epoch: {} {}/{}".format(epoch,i,total_step))
    
        # train
        correct = 0                                 # ne kadar doğru bildiğimizi tutuyoruz        
        total = 0                                   # ne kadar veri olduğunu tutuyoruz 
        with torch.no_grad():                       # no_grad ile train aşamasını bitiriyoruz
            for data in trainloader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                images = images.view(batch_size,1,64,32)
                images = images.float()
                
                # gpu
                if use_gpu:
                    if torch.cuda.is_available():
                        model.cuda()
                        images, labels = images.to(device), labels.to(device)
                        
                outputs = model(images)
                _, predicted = torch.max(outputs.data,1)                    # içerisinden max olanını bulup preddict ediyoruz
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy train %d %%"%(100*correct/total))
        train_acc.append(100*correct/total)
    
        # test
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.view(batch_size,1,64,32)
                images = images.float()
                
                # gpu
                if use_gpu:
                    if torch.cuda.is_available():
                        images, labels = images.to(device), labels.to(device)
                        
                outputs = model(images)
                _, predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
       
        print("Accuracy test %d %%"%(100*correct/total))
        test_acc.append(100*correct/total)
    
        loss_list.append(loss.item())
    
    #%% visualize
    
    fig, ax1 = plt.subplots()
    #plt.plot(test_acc,label = "acc",color = "black")
    #ax2 = ax1.twinx()
    ax1.plot(np.array(test_acc)/100,label = "Test Acc",color="green")
    #ax2.plot(np.array(train_acc)/100,label = "Train Acc",color= "red")
    ax1.legend()
    #ax2.legend()
    ax1.set_xlabel('Epoch')
    fig.tight_layout()
    plt.title("Test Accuracy")
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
                cv2.imwrite(r'C:\Users\oguzh\Desktop\kendim\%d'%i+".jpg",gray)
                plt.imshow(gray)
                i = i +1
                if i == 16:
                    i=0
                    break
        
        pngs = glob(r'C:\Users\oguzh\Desktop\kendim\*.jpg')       
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
            
            rs = cv2.resize(x , (32,64))
            gray = rgb2gray(rs)
            cv2.imwrite(j[:-3]+ 'jpg',gray)
            
        k_path = r"C:\Users\oguzh\Desktop\kendim"
        num_k_img = 16
        k_array = read_images(k_path,num_k_img)
        x_k_tensor = torch.from_numpy(k_array)
        
        y_ktrain_negative_tensor = torch.zeros(num_k_img,dtype = torch.long)
        
        
        ktest = torch.utils.data.TensorDataset(x_k_tensor)    
        dataiter = iter(ktest)
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.cpu(), 1)  
        x = 0
        y = 0  
        for i in range(16):
                if predicted[i] == 0:
                    x = x+1
                else:
                    y = y+1
        if x>y:
            print("Ortalama Uyanık ")
        else:
            print("Ortalama Yorgun")              
        print( ''.join('%16s' % classes[predicted[j]]
                                            for j in range(16)))
        i = 0
       