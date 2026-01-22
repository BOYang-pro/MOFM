# MOFM
Code implementation of "MOFM: A Multiple-in-One Flow Mamba for Unregistered Multi-Modal Image Fusion"

## Update
- [2025/8] All code will be open after acceptance.
- ....
## Framework

## Environment

We test the code on PyTorch 2.1.1 + CUDA 11.8.

1. Create a new conda environment
```
conda create -n MOFM python=3.10
conda activate MOFM
```

2. Install dependencies
```
conda install cudatoolkit==11.8
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install setuptools==68.2.2
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install nvidia/label/cuda-11.8.0::cuda-cudart-dev  
conda install nvidia/label/cuda-11.8.0::libcusparse-dev  
conda install nvidia/label/cuda-11.8.0::libcublas-dev  
conda install nvidia/label/cuda-11.8.0::libcusolver-dev  
cd kernels/selective_scan && pip install .
pip install -r requirements.txt
```
## Test
You can directly test our model to generate fused images using the following code (note: the pre-training weights sholud be saved in the './check/' file)

Link：

```
#Visible and infrared image fusion
python test-ir.py

#Visible and near-infrared image fusion
python test-nir.py

#Medical image fusion
python test-med.py

```
You can find their corresponding configuration file paths in './config/'.


## Train

### 1. Data download
The datasets are available for download at the following link. 

<div align="center">

|  Task  |  dataset Name  |  Link  |
| :----------: | :----------: | :-----: |
|         | M3FD   |  https://github.com/JinyuanLiu-CV/TarDAL  |
|  VI-IR  | RoadScene   |  https://github.com/hanna-xu/RoadScene  |
|         | LLVIP  |  https://github.com/bupt-ai-cz/LLVIP |
|         | FLIR_aligned  |https://huggingface.co/datasets/UserNae3/FLIR_aligned|
|  VI-NIR |  RGB-NIR Scene   |  https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/  |
|   |  MCubeS  |   https://github.com/kyotovision-public/multimodal-material-segmentation |
|  MED    |  Harvard  |  https://www.med.harvard.edu/AANLIB/home.html  ||
</div>

### 2. Unregistered data generation
This subsection provides tools to generate training and testing datasets with various types of image deformations, including **homography**, **affine**, **TPS**, **elastic**, and **fisheye** transformations. 

#### 2.1 Supported deformation types
<div align="center">

| Deformation Type | Description | Formula | Effect |
|:----------------:|:-----------|:------------|:------|
| **Affine** | Linear transformation: translation, rotation, scaling, shearing | $x' = A x + t$, where $A$ is a 2×2 matrix, $t$ is translation | Rigid smooth deformation |
| **Homography** | Single-plane projective transformation | $x' = \frac{H x}{h_3^T x}$, where $H$ is a 3×3 homography matrix | Rigid projective deformation |
| **TPS** | Thin Plate Spline | $x' =a + B x + \sum_i w_i U(x-c_i), \quad U(r) = r^2 \log r$ | Non-rigid deformation based on point control |
| **Elastic** | Local elastic distortions | $x' = x + \alpha d(x)$, where $d(x)$ is a random smooth displacement field | Subpixel-level local non-rigid deformation |
| **Fisheye** | Lens distortion simulating fisheye effect | $r' = f *\arctan(r/f)$ | Circular/curved non-rigid deformation |
</div>

#### 2.2 Generate datasets
Run the following command to generate datasets under **homography**, **affine**, or **elastic** deformations:
```
cd Data_generate

# Homography
python generate_warp_dataset.py --image_size 512 --transform_type hom

# Affine
python generate_warp_dataset.py --image_size 512 --transform_type affine

# Elastic
python generate_warp_dataset.py --image_size 512 --transform_type elastic
```
⚠ Adjust the parameters to control the strength and scale of the deformation.


Link： 

Run the following command to generate datasets under **TPS** or **Fisheye** deformations (Note that you need download Fisheye deformation files in the "./Data_generate/fisheye/flow/" or "./Data_generate/TPS/flow/":):
```
cd Data_generate 

python generate_Fisheye_dataset.py

python generate_tps_dataset.py
```

### 3 Data format
The data should organized in the following format:
```
train
├── VI-IR                                      
│   ├──ir                                      
│   │   ├──0000.png                            
...  
│   ├──ir_warp                                      
│   │   ├──0000.png                            
...                                           
│   ├──vi                                      
│   │   ├──0000.png                            
...                                           
│   ├──vi_warp                                       
│   │   ├──0000.png                            
...                                             


├── VI-NIR                                        
│   ├──nir                                   
│   │   ├──0000.png                            
...                                            
│   ├──nir_warp                                    
│   │   ├──0000.png  
│   ├──vi                                  
│   │   ├──0000.png                            
...                                            
│   ├──vi_warp                                    
│   │   ├──0000.png                           
...                                            


├── MED                                        
│   ├──mri                                    
│   │   ├──0000.png                            
...  
│   ├──mri_warp                                    
│   │   ├──0000.png                            
...                                           
│   ├──pet                                   
│   │   ├──0000.png                            
... 
│   ├──pet_warp                                   
│   │   ├──0000.png                            
... 

```
### 4. Start training
You can use the Dollowing code to train the LFDT-Fusion model for different fusion tasks.
```
torchrun  --nproc_per_node=3 --master_port=29600 train.py --task VI-NIR  --batch_size 14 --img_size 256  --net CrossBC
```
* nproc_per_node: This parameter represents the number of GPU. （Note: If you want to change nproc_per_node, the device number of "gpu_ids" in the configuration file './config/train.json' needs to be changed as well. For example, if nproc_per_node=2, gpu_ids=[0,1].）

* master_port: This parameter represents the port number used for communication by the master process.

* task: This parameter represents the selection of the fusion task. There are seven options: "MED, VI-IR, VI-NIR". 

* batch_size: This parameter represents the batch size.

* img_size: This parameter represents the image size.

* net: This parameter represents a model network, which defaults to a CrossBC network.

If you want to train three fusion tasks at once, you can also run the following code:

```
python sample_all.py
```

## Citation
```
@article{
    author    = {Bo Yang, Zhaohui Jiang, Dong Pan, Zhiping Lin, and Weihua Gui},
    title     = {MOFM: A Multiple-in-One Flow Mamba for Unregistered Multi-Modal Image Fusion},
    booktitle = {IEEE Transactions on Circuits and Systems for Video Technology},
    year      = {-},
    pages     = {-},
    doi       = {-},
}
```
