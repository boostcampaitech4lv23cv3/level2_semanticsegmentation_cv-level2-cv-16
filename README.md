# **Trash Semantic Segmentation**
![Main](https://user-images.githubusercontent.com/103131249/214734345-8c7eb577-127d-4e3c-8032-c1d03327f77f.png)

## ♻ **Contributors**

**CV-16조 💡 비전길잡이 💡**</br>NAVER Connect Foundation boostcamp AI Tech 4th

|민기|박민지|유영준|장지훈|최동혁|
|:----:|:----:|:----:|:---:|:---:|
|[<img alt="revanZX" src="https://avatars.githubusercontent.com/u/25689849?v=4&s=100" width="100">](https://github.com/revanZX)|[<img alt="arislid" src="https://avatars.githubusercontent.com/u/46767966?v=4&s=100" width="100">](https://github.com/arislid)|[<img alt="youngjun04" src="https://avatars.githubusercontent.com/u/113173095?v=4&s=100" width="100">](https://github.com/youngjun04)|[<img alt="FIN443" src="https://avatars.githubusercontent.com/u/70796031?v=4&s=100" width="100">](https://github.com/FIN443)|[<img alt="choipp" src="https://avatars.githubusercontent.com/u/103131249?v=4&s=117" width="100">](https://github.com/choipp)|
|CVAT</br>Stratified K-Fold</br>ConvNeXt| SeMask</br>Pseudo-labeling</br>Ensemble | HorNet · Swin</br>Optimization</br>Data version test | ViT-Adapter</br>Data split · merge</br>WandB customization| EVA · DiNAT</br>Class weights</br>Annotation manual|
|***Data</br>Cleansing***|***Data</br>Cleansing***|***Data</br>Cleansing***|***Data</br>Cleansing***|***Data</br>Cleansing***|
</br>

## ♻ **Links**

- [비전 길잡이 Notion 📝](https://vision-pathfinder.notion.site/Segmentation-3149d54760e1403c84ba094d7735a2af)
- [Annotation Tool - CVAT 매뉴얼](https://iot-meets-ai.notion.site/CVAT-516e44b823f34280aed3b50d4aaebcab)

## ♻ **Result**

![Result](https://user-images.githubusercontent.com/103131249/214524350-2d7bc75b-bb26-41a8-9f82-67841bbc68d9.png)

---

## ♻ **문제 정의**
- 대량 생산, 대량 소비로 인한 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제 발생
- 정확한 분리수거를 돕는 우수한 성능의 모델을 개발하는 목적의 프로젝트

## ♻ **Dataset**

![image](https://user-images.githubusercontent.com/113173095/214522741-32cbdcdd-2587-47c5-80c8-52b3c1866d3a.png)

- 학습 데이터 3,272장(train 2,617장, validation 655장) / 평가 데이터 819장
- 11개 클래스 : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic,
Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (512, 512)

## ♻ **Stratified Group K-Fold**

![k-fold](https://user-images.githubusercontent.com/113173095/214523254-3e2f8093-b4e8-4f13-876c-52c7b1289c73.png)

- 매우 불균형한 전체 train set의 클래스 분포
- 동일한 분포를 가지는 5개의 train, validation set 구성

## ♻ **Data Cleansing**

![Annotation Manual](https://user-images.githubusercontent.com/103131249/214528789-16cb5030-34b0-4d59-b65c-ee605fa9ebdd.png)

- 주어진 데이터의 경계선 annotation 오류 · 라벨링 일관성 부족 이슈
- **전체 데이터 3272장 전수조사 · 메뉴얼 작성 및 Data Cleansing 진행**
- **[CVAT](https://github.com/opencv/cvat) annotation tool** 사용 - 상단 Link 메뉴얼 참고
- **Data Versioning**
    - DataV1 : 수정 전 기본 데이터셋
    - DataV2 : Data Cleansing으로 1차 수정된 데이터셋
    - DataV3 : DataV2를 class별  area 기준으로 stratified k-fold 적용한 데이터셋

## ♻ **Model**

![image](https://user-images.githubusercontent.com/103131249/214537394-c87dbf48-7e6a-4886-b7a2-63a65def29ee.png)
| method | UperNet | SeMask | Mask2Former|
|:-:|:-:|:-:|:-:|
|model|Swin</br>Hornet</br>ConvNeXt</br>Beit(V1/V2)|Swin|Swin</br>Vit-adapter (beitV2)|

## ♻ **Experiments**

(1) 단일 모델 결과</br>
(2) k-fold ensemble 적용 결과

|      (1) Model       |  mIoU  |      (2) Model       |  mIoU  |
| :------------------: | :----: | :------------------: | :----: |
| upernet_hornet_large | 0.7310 | upernet_hornet_large | 0.7356 |
|        SeMask        | 0.7353 |        SeMask        | 0.7419 |
|   Mask2Former_Swin   | 0.7433 |   Mask2Former_Swin   | 0.7580 |
|     ViT-adapter      | 0.7418 |     ViT-adapter      | 0.7552 |


## ♻ **Ensemble**

|Swin-L|Adapter|SeMask|Adapter</br>dataV2|UperNet</br>Beit|Swin-L</br>dataV3|Hornet|Public</br>mIoU|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|✔︎|✔︎|✔︎|||||0.7716|
|✔︎|✔︎|✔︎|✔︎|✔︎|✔︎||0.7810|
|✔︎|✔︎|✔︎|✔︎|✔︎|✔︎|✔︎|**0.7828**|
- 다양한 모델들을 모두 포함시켰을 때 가장 높은 public mIoU 기록


## ♻ **Directory Structure**

```
|-- 🗂 appendix             : 발표자료 및 WrapUpReport
|-- 🗂 detection            : MMdet 기반 Deformable Attention 의존 코드 포함
|-- 🗂 mmsegmentation       : hornet, convnext, Beit 포함
|-- 🗂 segmentation         : mask2former_beitV2 adapter 학습
|-- 🗂 SeMask-Segmentation  : Detectron2 기반, mask2former_swin, Semask 학습
|-- 🗂 tools                : kfold 및 앙상블 등 자체 제작 툴 포함
`-- README.md
```

## ♻ **Installation**

### **nvidia-Apex**

```
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 22.05-dev
pip install -v --disable-pip-version-check --no-cache-dir ./
```

### **ViT-Adapter**

```
# Check CUDA & torch version
python -c 'import torch;print(torch.__version__)'
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Download mmcv-full==1.4.2 >> https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0 # for Mask2Former
pip install mmsegmentation==0.20.2
ln -s ../detection/ops ./

# If error occurred, check below context
cd ops & sh make.sh # compile deformable attention
```

### **Deformable DETR**

```
# Check CUDA version
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
sh cuda_11.0.2_450.51.05_linux.run

# If cv2 error occurred
apt-get install libgl1-mesa-glx

# Add this code to /root/.bashrc
export PATH="/usr/local/cuda-11.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH"

# Check CUDA & nvcc -V
source /root/.bashrc
python -c 'from torch.utils.cpp_extension import CUDA_HOME;print(CUDA_HOME)'

apt-get install g++

# Install MultiScaleDeformableAttention
cd ops
sh make.sh
```

### **Detectron2**

```
apt-get install ninja-build

conda create -n d2 python=3.8
source activate d2
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
python -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

# If error occurred (torch 1.7.0 requires dataclasses, which is not installed)
pip install dataclasses

# MultiScaleDeformableAttention 설치 방법은 Vit-adapter과 동일
git clone https://github.com/IDEA-Research/MaskDINO.git
cd MaskDINO
pip install -r requirements.txt
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
```
