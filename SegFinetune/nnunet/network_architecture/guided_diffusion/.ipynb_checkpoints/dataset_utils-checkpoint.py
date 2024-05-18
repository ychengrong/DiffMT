import os, pdb, operator
import torchio as tio
import json
import numpy as np
from tqdm import tqdm
import nibabel
from types import SimpleNamespace
from nibabel import Nifti1Header, Nifti1Image


def load_datasets(json_path):
    """加载json数据配置
    Args:
        json_path: json格式的数据配置文件绝对路径
    Returns:
        name: 数据集的名称
        training_subjects: 
        val_subjects: 
        test_subjects: 
    """
    fp = open(json_path, encoding="utf-8")
    config = json.load(fp)
    name = config['name']
    
    return config['name'], config['labels'], config['training'], config['val'], config['test']

def sizeconfuseRemove(
    subjects, 
    attrs = ["shape", "origin"],
    relative_tolerance = 1e-4, 
    absolute_tolerance = 1e-4,
    ):
    """
    剔除列表中image和label的shape不一致的item
    """
    removed_subjects, remove_list = [], []
    for subject in subjects:
        # d = abs(round(sum(np.array(subject['image'].origin) - np.array(subject['label'].origin)), 2))
        # all_close = 
        all_closes = [np.allclose(getattr(subject['image'], ac), getattr(subject['label'], ac), rtol=relative_tolerance, atol=absolute_tolerance) for ac in attrs]

        # if not operator.eq(subject['image'].shape, subject['label'].shape):
        #     remove_list.append(subject['pid'])
        # elif d > 1:
        if False in all_closes:
            remove_list.append(subject['pid'])
        else:
            removed_subjects.append(subject)

    return removed_subjects, remove_list

def load_subjects(json_path):
    """加载json数据配置
    Args:
        json_path: json格式的数据配置文件绝对路径
    Returns:
        name: 数据集的名称
        training_subjects: 
        val_subjects: 
        test_subjects: 
    """
    fp = open(json_path, encoding="utf-8")
    config = json.load(fp, object_hook=lambda d: SimpleNamespace(**d))
    name = config.name
    training_subjects, val_subjects, test_subjects = [], [], []
    for dataset in config.training:
        image_path, label_path = dataset.image, dataset.label
        subject = tio.Subject(
            pid = os.path.basename(os.path.dirname(image_path)),
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path))
        training_subjects.append(subject)
    
    for dataset in config.val:
        image_path, label_path = dataset.image, dataset.label
        subject = tio.Subject(
            pid = os.path.basename(os.path.dirname(image_path)),
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path))
        val_subjects.append(subject)
    
    for dataset in config.test:
        image_path, label_path = dataset.image, dataset.label
        subject = tio.Subject(
            pid = os.path.basename(os.path.dirname(image_path)),
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path))
        test_subjects.append(subject)
    
    return config.name, config.labels, training_subjects, val_subjects, test_subjects

def load_files_multi_channel(json_path):
    fp = open(json_path, encoding="utf-8")
    config = json.load(fp)
    test_files_1mm = [{"pid":list(tr.keys())[0],"image":  tr[list(tr.keys())[0]]['1mm']['image'], "label": tr[list(tr.keys())[0]]['1mm']['label']} for tr in config["test"]]
    test_files_3mm = [{"pid":list(tr.keys())[0],"image":  tr[list(tr.keys())[0]]['3mm']['image'], "label": tr[list(tr.keys())[0]]['3mm']['label']} for tr in config["test"]]
    test_files_5mm = [{"pid":list(tr.keys())[0],"image":  tr[list(tr.keys())[0]]['5mm']['image'], "label": tr[list(tr.keys())[0]]['5mm']['label']} for tr in config["test"]]
    return config['name'], config['labels'], config["training"], config["val"], test_files_1mm, test_files_3mm, test_files_5mm

def load_files_multi(json_path):
    fp = open(json_path, encoding="utf-8")
    config = json.load(fp)
    train_files = [{
        "pid":list(tr.keys())[0], 
        # "image_1mm": tr[list(tr.keys())[0]]['1mm']['image'], 
        # "label_1mm": tr[list(tr.keys())[0]]['1mm']['label'],
        "image": tr[list(tr.keys())[0]]['3mm']['image'],
        "label": tr[list(tr.keys())[0]]['3mm']['label'],
        # "image_5mm": tr[list(tr.keys())[0]]['5mm']['image'],
        # "label_5mm": tr[list(tr.keys())[0]]['5mm']['label']
        } for tr in config["training"]]
    val_files = [{"pid":list(tr.keys())[0], "image":  tr[list(tr.keys())[0]]['3mm']['image'], "label": tr[list(tr.keys())[0]]['3mm']['label']} for tr in config["val"]]
    test_files_1mm = [{"pid":list(tr.keys())[0],"image":  tr[list(tr.keys())[0]]['1mm']['image'], "label": tr[list(tr.keys())[0]]['1mm']['label']} for tr in config["test"]]
    test_files_3mm = [{"pid":list(tr.keys())[0],"image":  tr[list(tr.keys())[0]]['3mm']['image'], "label": tr[list(tr.keys())[0]]['3mm']['label']} for tr in config["test"]]
    test_files_5mm = [{"pid":list(tr.keys())[0],"image":  tr[list(tr.keys())[0]]['5mm']['image'], "label": tr[list(tr.keys())[0]]['5mm']['label']} for tr in config["test"]]
    return config['name'], config['labels'], train_files, val_files, test_files_1mm, test_files_3mm, test_files_5mm

def load_subjects_multi(json_path):
    """加载json数据配置
    Args:
        json_path: json格式的数据配置文件绝对路径
    Returns:
        name: 数据集的名称
        training_subjects: 
        val_subjects: 
        test_subjects: 
    """
    fp = open(json_path, encoding="utf-8")
    # config = json.load(fp, object_hook=lambda d: SimpleNamespace(**d))
    config = json.load(fp)
    training_subjects,val_subjects = [], []
    test_subjects_1mm, test_subjects_3mm, test_subjects_5mm = [], [], []

    training_set, val_set, test_set = config['training'], config['val'], config['test']

    for tr in training_set:
        # print(tr)
        data = tr[list(tr.keys())[0]]
        image_1mm, image_3mm, image_5mm = data['1mm']['image'], data['3mm']['image'], data['5mm']['image']
        label_1mm, label_3mm, label_5mm = data['1mm']['label'], data['3mm']['label'], data['5mm']['label']
        subject= tio.Subject(
                pid = os.path.basename(os.path.dirname(image_3mm)).split("_")[0],
                image_1mm=tio.ScalarImage(image_1mm),
                image_3mm=tio.ScalarImage(image_3mm),
                image_5mm=tio.ScalarImage(image_5mm),
                label_1mm=tio.LabelMap(label_1mm),
                label_3mm=tio.LabelMap(label_3mm),
                label_5mm=tio.LabelMap(label_5mm)
        )
        training_subjects.append(subject)

    for tr in val_set:
        data = tr[list(tr.keys())[0]]
        image_1mm, image_3mm, image_5mm = data['1mm']['image'], data['3mm']['image'], data['5mm']['image']
        label_1mm, label_3mm, label_5mm = data['1mm']['label'], data['3mm']['label'], data['5mm']['label']
        subject= tio.Subject(
                pid = os.path.basename(os.path.dirname(image_3mm)).split("_")[0],
                image_1mm=tio.ScalarImage(image_1mm),
                image_3mm=tio.ScalarImage(image_3mm),
                image_5mm=tio.ScalarImage(image_5mm),
                label_3mm=tio.LabelMap(label_3mm)
        )
        val_subjects.append(subject)

    for tr in test_set:
        data = tr[list(tr.keys())[0]]
        image_1mm, image_3mm, image_5mm = data['1mm']['image'], data['3mm']['image'], data['5mm']['image']
        label_1mm, label_3mm, label_5mm = data['1mm']['label'], data['3mm']['label'], data['5mm']['label']

        subject_1mm= tio.Subject(
                pid = os.path.basename(os.path.dirname(image_1mm)).split("_")[0],
                image_1mm=tio.ScalarImage(image_1mm),
                image_3mm=tio.ScalarImage(image_3mm),
                label_1mm=tio.LabelMap(label_1mm),
        )
        test_subjects_1mm.append(subject_1mm)

        subject_3mm= tio.Subject(
                pid = os.path.basename(os.path.dirname(image_3mm)).split("_")[0],
                image_3mm=tio.ScalarImage(image_3mm),
                label_3mm=tio.LabelMap(label_3mm),
        )
        test_subjects_3mm.append(subject_3mm)

        subject_5mm= tio.Subject(
                pid = os.path.basename(os.path.dirname(image_5mm)).split("_")[0],
                image_5mm=tio.ScalarImage(image_5mm),
                image_3mm=tio.ScalarImage(image_3mm),
                label_5mm=tio.LabelMap(label_5mm),
        )
        test_subjects_5mm.append(subject_5mm)
    
    return config['name'], config['labels'], training_subjects, val_subjects, test_subjects_1mm, test_subjects_3mm, test_subjects_5mm



def load_subjects_multi_image(json_path, target_multi = ["1mm", "5mm"]): #/root/worspace/data2/ycr_workspace/multiChannel/train
    training_multi = {}
    
    for thickness in target_multi:
        target_thickness = os.path.join(json_path, thickness)
        for sb_file in os.listdir(target_thickness):
            if not thickness in training_multi.keys():
                training_multi[thickness] = []
            target_subject = tio.Subject(
                image_stack = tio.ScalarImage(os.path.join(target_thickness, sb_file))
            )
            training_multi[thickness].append(target_subject)
                

    return training_multi