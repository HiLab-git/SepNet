# SepNet
code for Automatic Segmentation of Organs-at-Risk fromHead-and-Neck CT using Separable ConvolutionalNeural Network with Hard-Region-Weighted Loss, which won the third place of StructSeg19 task1. 
## Abstract
Nasopharyngeal Carcinoma (NPC) is a leading form of Head-and-Neck (HAN) cancer in the Arctic, China, Southeast Asia, and the Middle East/North Africa. Accurate segmentation of Organs-at-Risk (OAR) from Computed Tomography (CT) images with uncertainty information is critical for effective planning of radiation therapy for NPC treatment. Despite the state-of-the-art performance achieved by Convolutional Neural Networks (CNNs) for automatic segmentation of OARs, existing methods do not provide uncertainty estimation of the segmentation results for treatment planning, and their accuracy is still limited by several factors, including the low contrast of soft tissues in CT, highly imbalanced sizes of OARs and large inter-slice spacing. To address these problems, we propose a novel framework for accurate OAR segmentation with reliable uncertainty estimation. First, we propose a Segmental Linear Function (SLF) to transform the  intensity of CT images so that better visibility of different OARs is obtained to facilitate the segmentation task. Second, to deal with the large inter-slice spacing, we introduce a novel network (named as 3D-SepNet) based on spatially separated inter-slice convolution and intra-slice convolution. Thirdly, to deal with organs or regions that are hard to segment, we propose a hard voxel weighting strategy that automatically pays more attention to hard voxels for better segmentation. Finally, we use an ensemble of models trained with different loss functions and intensity transforms to obtain robust results, which also leads to segmentation uncertainty without extra efforts. Our method won the third place of the HAN OAR segmentation task in StructSeg 2019 challenge and it achieved weighted average Dice of 80.52% and 95% Hausdorff Distance of 3.043 mm. Experimental results show that 1) our SLF for intensity transform helps to improve the accuracy of OAR segmentation from CT images; 2) With only 1/3 parameters of 3D UNet, our 3D-SepNet obtains better segmentation results for most OARs; 3) The proposed hard voxel weighting strategy used for training effectively improves the segmentation accuracy; 4) The segmentation uncertainty obtained by our method has a high correlation to mis-segmentations, which has a potential to assist more informed decisions  in clinic practice.
![image](https://github.com/LWHYC/SepNet/blob/master/fig/summary.jpg)

## Requirements
Pytorch >= 1.4, SimpleITK >= 1.2, scipy >= 1.3.1, nibabel >= 2.5.0 and some common packages.

## Usages
Prepare StructSeg2019 task1 data and split them into two folders: train and valid. ( Each patient's CT image and label should be in a individual folder in train or valid folder) ;
Preprocess the data by `data_process/Preprocess.py`;

Change the `data_root` in `config/train.txt` to your data root;
Run `Python train.py`.
Your model is saved as 'model_save_prefix' in 'config/train.txt'.
