import os
import copy
import torch
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision
from torchvision.io import read_image
from torchvision import transforms
from torchvision import tv_tensors
from torchvision.transforms import v2
from pytorch_lightning import LightningModule, LightningDataModule
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.io import imread
from typing import Optional, Callable
from PIL import Image
import h5py

import zipfile
import urllib.request as urllib2

from data_handling.caching import SharedCache

class JSRTDataset(Dataset):
    def __init__(self, data: pd.DataFrame, data_dir: str, augmentation: bool = False):
        # download zip file from https://www.doc.ic.ac.uk/~bglocker/teaching/mli/JSRT.zip
        # and extract it to a directory

        self.data = data.reset_index(drop=True)
        self.data_dir = data_dir
        self.do_augment = augmentation

        # photometric data augmentation
        self.photometric_augment = transforms.Compose([
            transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        ])

        # geometric data augmentation
        self.geometric_augment = transforms.Compose([
            transforms.RandomApply(transforms=[
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.1), interpolation=transforms.InterpolationMode.NEAREST)
            ], p=0.2),
            transforms.RandomRotation(20, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply(transforms=[
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ], p=0.2),
        ])

        # collect samples (file paths) from dataset
        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = os.path.join(self.data_dir, 'images', self.data.loc[idx, 'study_id'])
            lab_path = os.path.join(self.data_dir, 'masks', self.data.loc[idx, 'study_id'])

            sample = {'image_path': img_path, 'labelmap_path': lab_path}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        # conver to torch tensors and add batch dimension
        image = torch.from_numpy(sample['image']).unsqueeze(0)
        labelmap = torch.from_numpy(sample['labelmap']).unsqueeze(0)

        # apply data augmentation
        if self.do_augment:
            image = self.photometric_augment(image.type(torch.ByteTensor)).type(torch.FloatTensor)

            # Stack the image and mask together so they get the same geometric transformations
            stacked = torch.cat([image, labelmap], dim=0)  # shape=(2xHxW)
            stacked = self.geometric_augment(stacked)

            # Split them back up again and convert labelmap back to datatype long
            image, labelmap = torch.chunk(stacked, chunks=2, dim=0)
            labelmap = labelmap.type(torch.LongTensor)

        # normalize image intensities to [0,1]
        image /= 255

        return {'image': image, 'labelmap': labelmap}

    def get_sample(self, item):
        sample = self.samples[item]

        # read image and labelmap from disk
        image = imread(sample['image_path']).astype(np.float32)
        labelmap = imread(sample['labelmap_path']).astype(np.int64)

        # convert labelmap to consecutive labels
        labelmap[labelmap==250] = 1 # heart
        labelmap[labelmap==200] = 2 # left lung
        labelmap[labelmap==150] = 3 # right lung
        labelmap[labelmap>3] = 0 # background

        return {'image': image, 'labelmap': labelmap}
    
class JSRTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = './data/JSRT/', batch_size: int = 5, num_workers: int = 4, augmentation: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # if jsrt_metadata.csv does not exist, download the dataset
        print(os.path.exists(os.path.join(self.data_dir, 'jsrt_metadata.csv')))
        if not os.path.exists(os.path.join(self.data_dir, 'jsrt_metadata.csv')):
            download_JRST(data_dir)

        self.data = pd.read_csv(os.path.join(self.data_dir, 'jsrt_metadata.csv'))
        dev, self.test_data = train_test_split(self.data, test_size=0.9)
        self.train_data, self.val_data = train_test_split(dev, test_size=0.9)

        self.train_set = JSRTDataset(self.train_data, self.data_dir, augmentation=augmentation)
        self.val_set = JSRTDataset(self.val_data, self.data_dir, augmentation=False)
        self.test_set = JSRTDataset(self.test_data, self.data_dir, augmentation=False)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
class CheXpertDataset(Dataset):
    def __init__(self, data: list, data_dir: str, augmentation: bool = False, cache: bool = False):
        # download zip file from https://www.doc.ic.ac.uk/~bglocker/teaching/mli/JSRT.zip
        # and extract it to a directory

        self.data = data
        self.data_dir = data_dir
        self.do_augment = augmentation
        self.cache = cache

        if cache:
            self.cache = SharedCache(
                size_limit_gib=24,
                dataset_len=len(self.data),
                data_dims=[1, 224, 224],
                dtype=torch.float32,
            )
        else:
            self.cache = None

        # photometric data augmentation
        self.photometric_augment = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(p=0.15),
            transforms.RandomVerticalFlip(p=0.15),

        ])

        # geometric data augmentation
        self.geometric_augment = transforms.Compose([
            transforms.RandomApply(transforms=[transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), interpolation=transforms.InterpolationMode.NEAREST)], 
                                                p = 0.5),
            transforms.RandomRotation(15),
            transforms.RandomApply(transforms=[transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.1))] , p=0.3),
            transforms.RandomHorizontalFlip(p=0.15), 
            transforms.RandomVerticalFlip(p=0.15),
        ])

        # collect samples (file paths) from dataset
        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(data)), desc='Loading Data')):

            sample = {'image_path': data[idx]}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        
        # if self.cache is not None:
        #     sample = self.cache.get_slot(item)
        #     if sample is None:
        #         sample = self.get_sample(item)
        #         self.cache.set_slot(item, sample, allow_overwrite=False)
        # else:
        sample = self.get_sample(item)

        # add batch dimension
        image = sample.unsqueeze(0)

        # apply data augmentation
        if self.do_augment:
            #image = self.photometric_augment(image.type(torch.ByteTensor)).type(torch.FloatTensor)
            image = self.geometric_augment(image.type(torch.ByteTensor)).type(torch.FloatTensor)
            # Stack the image and mask together so they get the same geometric transformations
            # stacked = torch.cat([image], dim=0)  # shape=(2xHxW)
            # stacked = self.geometric_augment(stacked)

            # Split them back up again and convert labelmap back to datatype long
            # image = torch.chunk(stacked, chunks=2, dim=0)

        # normalize image intensities to [0,1]
        image /= 255

        return {'image': image}

    def get_sample(self, item):
        sample = self.samples[item]
        # read image and labelmap from disk
        image = imread(sample['image_path']).astype(np.float32)
        image = torch.from_numpy(image).float()

        return image#{'image': image}
    
class CheXpertDataModule(LightningDataModule):
    def __init__(self, data_dir: str = 'vol/biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224/', batch_size: int = 32, num_workers: int = 6, cache: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # if jsrt_metadata.csv does not exist, download the dataset
        # if not os.path.exists(os.path.join(self.data_dir, 'jsrt_metadata.csv')):
        #     download_JRST(data_dir)

        files = os.listdir(data_dir)      # pd.read_csv(os.path.join(self.data_dir, 'data_incl_224.csv'))
        self.data = [os.path.join(data_dir, file) for file in files]
        dev, self.test_data = train_test_split(self.data, test_size=0.2)
        self.train_data, self.val_data = train_test_split(dev, test_size=0.05)

        self.train_set = CheXpertDataset(self.train_data, self.data_dir, augmentation=True, cache=cache)
        self.val_set = CheXpertDataset(self.val_data, self.data_dir, augmentation=False, cache=cache)
        self.test_set = CheXpertDataset(self.test_data, self.data_dir, augmentation=False, cache=cache)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    

class CLEVRNDataset(LightningDataModule):
    def __init__(self, data_dir: str = '/vol/bitbucket/bc1623/project/semi_supervised_uncertainty/data/clevr', input_res: int = 224, max_num_obj: int = 6, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.input_res = input_res
        self.max_num_obj = max_num_obj
        self.batch_size = batch_size

        n = self.input_res * 0.004296875

        args = {'input_res': input_res, 'max_num_obj': max_num_obj, 'batch_size': batch_size, 'data_dir': data_dir}

        n = args['input_res'] * 0.004296875  # equiv 0.55 for 128
        h, w = int(320 * n), int(480 * n)
        aug = {
            "train": transforms.Compose(
                [
                    transforms.Resize((h, w), antialias=None),
                    # transforms.CenterCrop(args['input_res]),
                    transforms.RandomCrop(args['input_res']),
                    transforms.PILToTensor(),  # (0,255)
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize((h, w), antialias=None),
                    transforms.CenterCrop(args['input_res']),
                    transforms.PILToTensor(),  # (0,255)
                ]
            ),
        }

        datasets = {
            split: CLEVRN(
                torchvision.datasets.CLEVRClassification(
                    root=args['data_dir'],
                    split=split,# download=True,
                ),
                
                num_obj=args['max_num_obj'],
                transform=aug[split],
            )
            for split in ["train", "val"]
        }
        datasets["test"] = copy.deepcopy(datasets["val"])

        kwargs = {
            "batch_size": args['batch_size'],
            "num_workers": 6,#os.cpu_count(),  # 4 cores to spare
            "pin_memory": True,
        }

        self.dataloaders = {
            split: DataLoader(
                datasets[split],
                shuffle=(split == "train"),
                drop_last=(split == "train"),
                **kwargs,
            )
            for split in ["train", "val", "test"]
        }
        
    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        return self.dataloaders["val"]

    def test_dataloader(self):
        return self.dataloaders["test"]

class CLEVRN(Dataset):
    def __init__(
        self,
        clevr: Dataset,
        num_obj: int = 6,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache: bool = True,
    ):
        super().__init__()
        self.cache = cache
        assert num_obj >= 3 and num_obj <= 10
        assert clevr._split != "test"  # test set labels are None
        self.filter_idx = [i for i, y in enumerate(clevr._labels) if y <= num_obj]
        self._image_files = [clevr._image_files[i] for i in self.filter_idx]
        self._labels = [clevr._labels[i] for i in self.filter_idx]
        self.transform = transform
        self.target_transform = target_transform

        if self.cache:
            from concurrent.futures import ThreadPoolExecutor

            self._images = []
            with ThreadPoolExecutor() as executor:
                self._images = list(
                    tqdm(
                        executor.map(self._load_image, self._image_files),
                        total=len(self._image_files),
                        desc=f"Caching CLEVR {clevr._split}",
                        mininterval=10,
                    )
                )

    def _load_image(self, file):
        return Image.open(file).convert("RGB")

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, idx: int):
        if self.cache:
            image = self._images[idx]
        else:
            image = self._load_image(self._image_files[idx])
        label = self._labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return {'image': image, 'labelmap': label}

class LIDCDataGenerator(Dataset):

    def __init__(self, 
                 data_dir: str = "/data2/rmehta3/datasets/LIDC/LIDC_PNG/", # path to h5py file
                 augment: bool = False, # data augmentation if true
                 scale_range: float = 0.0, # if not 0, then random scaling in range of 1 +- scale_range   
                 rotation_degree: float = 0.0, # if not 0, then random rotation in range of +- rotation_degree
                 subset: str = "test" # dataset subtype, "test", "val", or "train"
                 ):
        
        # this h5py file contains already normalized "images", [0,1]
        #self.data = h5py.File(h5py_path, 'r')

        files = os.listdir(data_dir + '/Image/' + subset + '/')      # pd.read_csv(os.path.join(self.data_dir, 'data_incl_224.csv'))
        self.data = [os.path.join(data_dir + '/Image/' + subset + '/', file) for file in files]

        # masks
        mask_0 = os.listdir(data_dir + 'Masks/Mask_00/all/' + subset + '/')    
        self.mask_0 = [os.path.join(data_dir + 'Masks/Mask_00/all/' + subset + '/', file) for file in mask_0]  
        mask_1 = os.listdir(data_dir + 'Masks/Mask_01/all/' + subset + '/')
        self.mask_1 = [os.path.join(data_dir + 'Masks/Mask_01/all/' + subset + '/', file) for file in mask_1]
        mask_2 = os.listdir(data_dir + 'Masks/Mask_02/all/' + subset + '/')
        self.mask_2 = [os.path.join(data_dir + 'Masks/Mask_02/all/' + subset + '/', file) for file in mask_2]
        mask_3 = os.listdir(data_dir + 'Masks/Mask_03/all/' + subset + '/')
        self.mask_3 = [os.path.join(data_dir + 'Masks/Mask_03/all/' + subset + '/', file) for file in mask_3]
        
        self.augment = augment
        self.scale_range = scale_range
        self.rotation_degree = rotation_degree
        self.subset = subset

        # store indices of all subset images and labels
        #self.list_ids = [i for i in range(0, self.data[subset]['images'].shape[0])]


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
           
        # Find list of IDs
        X = read_image(self.data[idx])
        mask_0 = read_image(self.mask_0[idx])
        mask_1 = read_image(self.mask_1[idx])
        mask_2 = read_image(self.mask_2[idx])
        mask_3 = read_image(self.mask_3[idx])
        y = torch.stack([mask_0, mask_1, mask_2, mask_3], dim=0) # shape: (4,128,128) where 4 is the possible annotations, these are 4 binary annotations provided by 4 different raters

        # convert x to float32 y to uint8
        X_t = X.type(torch.float32)
        # normalise x to [0,1]
        X_t /= 255
        y_t = (y > 128).type(torch.float32)

        # perform dataaugmentation: dataaugmentations similar to https://github.com/baumgach/PHiSeg-code/blob/master/data/batch_provider.py#L140        
        if self.subset=="train" and self.augment:
            transforms = v2.Compose([
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=self.rotation_degree),
                v2.RandomResizedCrop(size=(X.shape[1],X.shape[2]),scale=(1-self.scale_range,1+self.scale_range))
                ])
            
            X_t, y_t = transforms(X_t, y_t)

        return {'image':X_t, 'labelmap':y_t} 
    
class LIDCDataModule(LightningDataModule):
    def __init__(self, data_dir: str = 'vol/biodata/data/chest_xray/CheXpert-v1.0/preproc_224x224/', batch_size: int = 32, num_workers: int = 6, cache: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = LIDCDataGenerator(augment=True, scale_range=0.1, rotation_degree=10, subset="train")
        self.val_set = LIDCDataGenerator(augment=False, subset="val")
        self.test_set = LIDCDataGenerator(augment=False, subset="test")

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    

def download_JRST(data_dir: str = './data/JSRT/'):
    data_url = "https://www.doc.ic.ac.uk/~bglocker/teaching/mli/JSRT.zip"

    print("Downloading ", data_url)
    response = urllib2.urlopen(data_url)
    zippedData = response.read()

    # save data to disk
    print("Saving to ", data_dir)
    output = open(data_dir + 'JSRT.zip','wb')
    output.write(zippedData)
    output.close()

    zfobj = zipfile.ZipFile(data_dir + 'JSRT.zip')
    #print(zfobj.namelist())
    for name in zfobj.namelist():
        # Get the directory name from the file path
        if '.png' not in name and '.csv' not in name:
            continue

        dir_name = os.path.dirname(name)
        # Create the directory if it doesn't exist
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        # Extract the file
        uncompressed = zfobj.read(name)
        
        # Save uncompressed data to disk
        file_path = name
        print("Saving extracted file to", file_path)
        output = open(file_path, 'wb')
        output.write(uncompressed)
        output.close()

    print("Data downloaded and extracted to ", data_dir)