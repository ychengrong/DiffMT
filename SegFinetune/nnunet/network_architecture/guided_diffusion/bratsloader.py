import torch
import torch.nn
import numpy as np
import os, json
import os.path
import nibabel
import torchvision.utils as vutils
from types import SimpleNamespace

from .dataset_utils import load_subjects_multi, load_subjects_multi_image


class TempleLobeDataset(torch.utils.data.Dataset):
    def __init__(self, config_path, transform_im, transform_la, test=False):
        super().__init__()
        fp = open(config_path, encoding="utf-8") #yu
        config = json.load(fp, object_hook=lambda d: SimpleNamespace(**d))
        dataset_name, label_dict, training_subjects, val_subjects, test_subjects_1mm, test_subjects_3mm, test_subjects_5mm = load_subjects_multi(config.dataset)

        self.transform_im = transform_im
        self.transform_la = transform_la
        self.test= test
        self.tr_sbs = None
        if self.test:
            self.tr_sbs = test_subjects_3mm
        else:
            self.tr_sbs =  training_subjects
            
        # self.tr_sbs = self.tr_sbs[:2]
        # self.ts_1mm_sbs = test_subjects_1mm
        # self.ts_3mm_sbs = test_subjects_3mm
        # self.ts_5mm_sbs = test_subjects_5mm
        

        self.seqtypes = ['image_1mm', 'image_3mm', 'image_5mm', 'label_3mm']
        self.seqtypes_tr = ['image_3mm', 'label_3mm']
        self.len_list = []
        
        print("Length of samples: {}".format(len(self.tr_sbs)))
        for sb in self.tr_sbs:
            self.len_list.append(sb[self.seqtypes[-1]].shape[-1])
        
        # for sb in training_subjects:
        #     for seqtype in self.seqtypes:
        #         if not seqtype in self.database.keys():
        #             self.database[seqtype] = []
        #         self.database[seqtype].append(sb[seqtype]['data'])
        
    def __getitem__(self, idx):
        out = []
        index = idx
        sample_index = 0
        for le in self.len_list:
            if index <= le:
                break
            index -= le
            sample_index += 1
        
        database = self.tr_sbs[sample_index]
        for setr in self.seqtypes_tr: 
            out.append(database[setr]['data'][..., index-1])
        out = torch.stack(out)
        image = out[:-1, ...]
        label = out[-1, ...][None, ...]
        if self.transform_im:
            state = torch.get_rng_state()
            image = self.transform_im(image)
            torch.set_rng_state(state)
            label = self.transform_la(label)
        image = image[0]
        label = label[0]
        return (image, label, "{}_{}".format(sample_index, index))

    def __len__(self):
        return sum(self.len_list)
    
    
class TempleLobeMultiDataset(torch.utils.data.Dataset):
    def __init__(self, config_path, transform, test=False, thickness="5mm"):
        super().__init__()
        train_subjects = load_subjects_multi_image(config_path)#"/root/workspace/data2/ycr_workspace/multiChannel/train/"

        self.transform = transform
        self.test= test
        self.tr_sbs = None
        if self.test:
            self.tr_sbs = train_subjects[thickness]
        else:
            self.tr_sbs = train_subjects[thickness]
        

        self.thickness = thickness
        
        self.len_list = []
        
        print("Length of samples: {}".format(len(self.tr_sbs)))
        for sb in self.tr_sbs:
            self.len_list.append(sb['image_stack'].shape[0])
        
    def __getitem__(self, idx):
        index = idx
        sample_index = 0
        for le in self.len_list:
            if index <= le:
                break
            index -= le
            sample_index += 1
        
        database = self.tr_sbs[sample_index]
        out = database['image_stack']['data'][index-1]
        label = out[:-1, ...]
        image = out[-1, ...][None, ...]
        # print(image.size(), label.size())
        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image[None, ...])
            torch.set_rng_state(state)
            label = self.transform(label[None, ...])
        image = image[0]
        label = label[0]
        return (image, label, "{}_{}".format(sample_index, index))

    def __len__(self):
        return sum(self.len_list)

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path)

    def __len__(self):
        return len(self.database)

class BRATSDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3].split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                if not set(datapoint.keys()) == self.seqtypes_set:
                    print(self.seqtypes_set)
                    print(set(datapoint.keys()))
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
    
    def __len__(self):
        return len(self.database) * 155

    def __getitem__(self, x):
        out = []
        n = x // 155
        slice = x % 155
        filedict = self.database[n]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            o = torch.tensor(nib_img.get_fdata())[:,:,slice]
            # if seqtype != 'seg':
            #     o = o / o.max()
            out.append(o)
        out = torch.stack(out)
        if self.test_flag:
            image=out
            # image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            # image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            # label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path

    def __len__(self):
        return len(self.database)


