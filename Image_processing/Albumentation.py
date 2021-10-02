import cv2
from torchvision.transforms import ToPILImage
import time
from torch.utils.data import Dataset
import albumentations.pytorch
import torch.nn.functional as F
import glob
import glob,os
from glob import glob



albumentations_transform_oneof = albumentations.Compose([
    # albumentations.Resize(256, 256), 
    # albumentations.RandomCrop(224, 224),
    albumentations.OneOf([                                                                      
                          albumentations.VerticalFlip(p=1),
                                
    ], p=1),
    albumentations.OneOf([
                          # albumentations.MotionBlur(p=1),
                          # albumentations.OpticalDistortion(p=1),
                          # albumentations.GaussNoise(p=1)                 
    ], p=1),
    albumentations.pytorch.ToTensor()
])

class AlbumentationsDataset(Dataset):
   
    def __init__(self, file_paths, labels, transform=None):
        print(file_paths)
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        image = cv2.imread(file_path)

        start_t = time.time()
        if self.transform:
            augmented = self.transform(image=image) 
            image = augmented['image']
            total_time = (time.time() - start_t)
        return image, label, total_time
pngs = glob(r'C:\Users\oguzh\Desktop\tez\*.jpg')
files = []  
i = 0      
for j in pngs:
    
    print(j[:-3]+'jpg')
    a = [j[:-3]+'jpg']
    print(a)
   
    albumentations_dataset = AlbumentationsDataset(
    
    file_paths=a,
    labels=[1],
    transform=albumentations_transform_oneof,
    )

    print(type(albumentations_dataset[0][0]))
    out = F.interpolate(albumentations_dataset[0][0],size=207)  
    ToPILImage()(out).save(r'C:\Users\oguzh\Desktop\a%d'%i+'.jpg', mode='jpg')
    i = i + 1
    

# albumentations_dataset = AlbumentationsDataset(
#     file_paths=[r"C:\Users\oguzh\Desktop\acu\1.jpg"],
#     labels=[1],
#     transform=albumentations_transform_oneof,
# )

# num_samples = 5
# fig, ax = plt.subplots(1, num_samples, figsize=(25, 5))
# for i in range(num_samples):
#   ax[i].imshow(transforms.ToPILImage()(albumentations_dataset[0][0]))
#   ax[i].axis('off')
# a = transforms.ToPILImage(albumentations_dataset[0][0])
# print(type(a))

# a.save(a,'aa.jpg')
# num_samples = 2
# fig, ax = plt.subplots(1, num_samples, figsize=(32, 64))
# for i in range(num_samples):
#   ax[i].imshow(transforms.ToPILImage()(albumentations_dataset[0][0]))   
#   # save_image(ax,'aa.jpg')
  
#   ax[i].axis('off')