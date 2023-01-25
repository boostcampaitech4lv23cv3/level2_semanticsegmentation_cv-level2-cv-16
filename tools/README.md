# **Trash Semantic Segmentation**

![Main](https://user-images.githubusercontent.com/103131249/214512646-bd6acd0d-17e6-4884-9204-cce8585bcb71.png)

## 🚮 **Contributors**

**CV-16조 💡 비전길잡이 💡**</br>NAVER Connect Foundation boostcamp AI Tech 4th

|민기|박민지|유영준|장지훈|최동혁|
|:----:|:----:|:----:|:---:|:---:|
|[<img alt="revanZX" src="https://avatars.githubusercontent.com/u/25689849?v=4&s=100" width="100">](https://github.com/revanZX)|[<img alt="arislid" src="https://avatars.githubusercontent.com/u/46767966?v=4&s=100" width="100">](https://github.com/arislid)|[<img alt="youngjun04" src="https://avatars.githubusercontent.com/u/113173095?v=4&s=100" width="100">](https://github.com/youngjun04)|[<img alt="FIN443" src="https://avatars.githubusercontent.com/u/70796031?v=4&s=100" width="100">](https://github.com/FIN443)|[<img alt="choipp" src="https://avatars.githubusercontent.com/u/103131249?v=4&s=117" width="100">](https://github.com/choipp)|
|CVAT 구현</br>Stratified K-Fold</br>convNeXt| SeMask</br>pseudo-labeling</br>Ensemble test | HorNet, Swin</br>Optimization</br>Data version test | ViT-Adapter</br>Data split, merge 구현</br>wandb logger 커스텀| EVA, DiNAT</br>class weights</br>annotation manual|
|Data Annotation    |||||

</br>


## 🚮 **Links**

- [비전 길잡이 Notion 📝](https://vision-pathfinder.notion.site/Segmentation-3149d54760e1403c84ba094d7735a2af)
- [비전 길잡이 발표자료 & WrapUpReport](./appendix/)
- [CVAT 구현](https://iot-meets-ai.notion.site/CVAT-516e44b823f34280aed3b50d4aaebcab)

## 🚮 **Result**
![Result](https://user-images.githubusercontent.com/103131249/214524350-2d7bc75b-bb26-41a8-9f82-67841bbc68d9.png)

---

## 🚮 **문제 정의**
![image](https://user-images.githubusercontent.com/70796031/214523079-7066fa79-d8c8-449a-b46a-2df376f67a65.png)
- 현재 대량 생산, 대량 소비의 시대에서 많은 물건이 대량으로 생산으로 인한 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제 발생
- 우수한 성능의 모델을 만들어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용하는 것이 목적

## 🚮 **Dataset**
![image](https://user-images.githubusercontent.com/113173095/214522741-32cbdcdd-2587-47c5-80c8-52b3c1866d3a.png)
- 학습 데이터 3,272장 (train 2617장, validation 655장), 평가 데이터 819장
- 11개 클래스 : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic,
Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (512, 512)

### Stratified Group K-Fold
![k-fold](https://user-images.githubusercontent.com/113173095/214523254-3e2f8093-b4e8-4f13-876c-52c7b1289c73.png)
- 전체 train set의 클래스 분포 매우 불균형
- 동일한 분포를 가지는 5쌍의 train, validation set 구성

## 🚮 **Data Cleansing**
Annotation 데이터의 경계선이 잘못 되거나 라벨링의 일관성이 부족하다고 판단하여 
### case 1. 잘못된 labeling

<img src="https://user-images.githubusercontent.com/46767966/214523896-9884eb65-1b5e-48a1-bcf6-63e365c1fdd6.png" width="300" height="300">

### case 2. 잘못된 annotation

<img src="https://user-images.githubusercontent.com/46767966/214525567-da6530dc-e983-4532-a14e-28beeb98575f.png" width="300" height="300">

## 🚮 **Model**

### UperNet

### Mask2Former

### ViT-Adapter

## 🚮 **LB Timeline ⌛**



- 초반에 ICDAR 17, 19 적용하여 높은 점수 확보
- SynthText 적용 후 ImageNet pretrained Backbone + 대량의 합성 데이터 pretrain
- 최종적으로 fine-tuning 통해 후반부에 성능 끌어올림

## 🚮 **Directory Structure**

```
|-- 🗂 appendix             : 발표자료 및 WrapUpReport
|-- 🗂 detection            : MMdet 기반 Deformable Attention 의존 코드 포함
|-- 🗂 mmsegmentation       : hornet, convnext, Beit 포함
|-- 🗂 segmentation         : mask2former_beit adapter 학습
|-- 🗂 SeMask-Segmentation  : Detectron2 기반, mask2former_swin, Semask 학습
|-- 🗂 tools                : kfold 및 앙상블 등 자체 제작 툴 포함
`-- README.md
```