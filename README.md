

# DiffMT
DiffMT is proposed to utilize DDPM-based pre-training for multi-thickness CT segmentation. This method is elaborated in our paper, "Leveraging Denoising Diffusion Probabilistic Model to Improve Multi-thickness CT Segmentation".

## Requirement
     Python 3.8.5+
     Pytorch 1.10.1+
     nnunet v1/v2
## Usage
### DDPMPretrain
The codes in `DDPMPretrain` is the implementation of the DDPM-based pretraining on multi-thickness CT generation.
1. Prepared your multi-thickness CT data, and your dataset folder under "data_dir" should be like:

~~~
dataset
└───1mm
│   │   image_13_1.nii.gz
│   │   image_13_2.nii.gz
│   │   ...
└───5mm
│   │   image_53_1.nii.gz
│   │   image_53_2.nii.gz
│   │   ...
~~~
    Specially, `xx` in each `image_xx_1.nii.gz` represents two different thicknesses CT, which stored with two-channel format.

    
2. For training, using the script in `DDPMPretrain/train.sh`, in which config your own `data_name`, `data_dir` and `out_dir`
    
3. (optional) For sampling, using the script in `DDPMPretrain/sample.sh`


### SegFinetune
The code in SegFinetune implements the fine-tuning process for multi-thickness CT segmentation. This code is designed based on the workflow of the  [nnunet](https://github.com/MIC-DKFZ/nnUNet) framework.

1. Data preparation follows the nnUNet methodology, with detailed implementation referring to [Dataset conversion](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1?tab=readme-ov-file#dataset-conversion) and [Experiment planning and preprocessing](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1?tab=readme-ov-file#experiment-planning-and-preprocessing)

2. Configure the path for the DDPM-based pre-training model and other training configurations as referenced in `SegFinetune/nnunet/training/network_training/nnUNetTrainerV2.py` with key word `if_use_checkpoint`

3. Run training with `nnUNet_train 2d nnUNetTrainerV2 10 0`

4. Run evaluation with `nnUNet_predict -i $inputDir -o $outputDir -t 10 -m 2d -tr nnUNetTrainerV2` and `nnUNet_evaluate_folder -pred $outputDir -ref $refDir -l 1`

5. More architectures can be implemented as `nnUNetTrainerV2_xxx.py` in `SegFinetune/nnunet/training/network_training`
    
