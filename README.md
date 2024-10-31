# DPTrack

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![test](https://img.shields.io/static/v1?label=By&message=Pytorch&color=red)

#### DPTrack is a simply and strong multi-object tracker. 



**DPTrack: A Density Perception-based Multi-Object Tracking Algorithm for Dense and Occluded Scenarios**


## Abstract

Multi-object tracking (MOT) is a critical research area in computer vision, particularly challenging in dense and occluded scenes. This paper proposes DPTrack, a density perception-based MOT algorithm designed to address these challenges. DPTrack introduces three key components: Density Graded Matching (DGM) for scene decomposition and accurate object association, Tracklet Offset Modeling (TOM) to enhance position and motion state assessment using relative positional offsets, and Region Proposal Strategy (RPS) for precise tracking region localization. By dividing dense object sets into sparse subsets and combining strong and weak cues, DPTrack demonstrates improved performance in handling complex scenarios. Experimental results on the MOT17, MOT20, and DanceTrack datasets show that DPTrack outperforms existing heuristic trackers across various metrics, proving its effectiveness in dense and occluded environments. Our method is implemented in a plug-and-play manner, requiring no additional training, and can be easily integrated into various advanced tracking algorithms. 

### Highlights

- DPTrack is an **outstanding** heuristic trackers on MOT17/MOT20 datasets, and performs excellently on DanceTrack, especially in dense and occluded environments. 
- Maintains **Simple, Online and Real-Time (SORT)** characteristics.
- **Training-free** and **plug-and-play** manner.
- **Strong generalization** for diverse trackers and scenarios

### Pipeline

<center>
<img src="a_images/3.1.jpg" width="800"/>
</center>

## News

* [10/31/2024]: The DPTrack paper is submitted to the Springer journal Visual Computer and is currently awaiting review.
* [10/30/2024]: DPTrack is supported in [yolo_tracking](https://github.com/mikel-brostrom/yolo_tracking). Many thanks to [@mikel-brostrom](https://github.com/mikel-brostrom) for the contribution.
* [10/30/2024]: DPTrack’s code is officially open-sourced. Visit (https://github.com/OOHany/DP-Track) for code and resources! Please note that our overall code is still being refined.

## Tracking performance

### Results on MOT17/MOT20 challenge test set

| Dataset          | HOTA | MOTA | IDF1 | AssA |  FP  |  FN  |  IDs |
| :--------------- | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| MOT17            | 64.0 | 79.9 | 79.2 | 63.6 | 28344| 84009| 1098 |
| MOT20            | 64.0 | 76.8 | 78.6 | 63.7 | 30383| 88828| 1059 |

### Results on DanceTrack test set

| Dataset          | HOTA | MOTA | IDF1 | AssA |
| :--------------- | :--: | :--: | :--: | :--: |
| DanceTrack       | 65.0 | 91.7 | 67.1 | 51.6 |


## Installation

DPTrack code is based on [OC-SORT](https://github.com/noahcao/OC_SORT) and [FastReID](https://github.com/JDAI-CV/fast-reid). The ReID component is optional and based on [FastReID](https://github.com/JDAI-CV/fast-reid). Tested the code with Python 3.8 + Pytorch 1.10.0 + torchvision 0.11.0.

Step1. Install DPTrack

```shell
git clone https://github.com/OOHany/DPTrack.git
cd DPTrack
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others

```shell
pip3 install cython_bbox pandas xmltodict
```

Step4. [optional] FastReID Installation

You can refer to [FastReID Installation](https://github.com/JDAI-CV/fast-reid/blob/master/INSTALL.md).

```shell
pip install -r fast_reid/docs/requirements.txt
```

## Data preparation

**Our data structure is the same as [OC-SORT](https://github.com/noahcao/OC_SORT).** 

1. Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [DanceTrack](https://github.com/DanceTrack/DanceTrack), [CUHKSYSU](http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html) and put them under <DPTrack_HOME>/datasets in the following structure (CrowdHuman, Cityperson and ETHZ are not needed if you download YOLOX weights from [ByteTrack](https://github.com/ifzhang/ByteTrack) or [OC-SORT](https://github.com/noahcao/OC_SORT)) :

   ```
   datasets
   |——————mot
   |        └——————train
   |        └——————test
   └——————crowdhuman
   |        └——————Crowdhuman_train
   |        └——————Crowdhuman_val
   |        └——————annotation_train.odgt
   |        └——————annotation_val.odgt
   └——————MOT20
   |        └——————train
   |        └——————test
   └——————Cityscapes
   |        └——————images
   |        └——————labels_with_ids
   └——————ETHZ
   |        └——————eth01
   |        └——————...
   |        └——————eth07
   └——————CUHKSYSU
   |        └——————images
   |        └——————labels_with_ids
   └——————dancetrack        
            └——————train
               └——————train_seqmap.txt
            └——————val
               └——————val_seqmap.txt
            └——————test
               └——————test_seqmap.txt
   ```

2. Prepare DanceTrack dataset:

   ```python
   # replace "dance" with ethz/mot17/mot20/crowdhuman/cityperson/cuhk for others
   python3 tools/convert_dance_to_coco.py 
   ```

3. Prepare MOT17/MOT20 dataset. 

   ```python
   # build mixed training sets for MOT17 and MOT20 
   python3 tools/mix_data_{ablation/mot17/mot20}.py
   ```

4. [optional] Prepare ReID datasets:

   ```
   cd <DPTrack_HOME>
   
   # For MOT17 
   python3 fast_reid/datasets/generate_mot_patches.py --data_path <dataets_dir> --mot 17
   
   # For MOT20
   python3 fast_reid/datasets/generate_mot_patches.py --data_path <dataets_dir> --mot 20
   
   # For DanceTrack
   python3 fast_reid/datasets/generate_cuhksysu_dance_patches.py --data_path <dataets_dir> 
   ```

## Model Zoo

Download and store the trained models in 'pretrained' folder as follow:

```
<DPTrack_HOME>/pretrained
```

### Detection Model

We provide some pretrained YOLO-X weights for DPTrack, which are inherited from [ByteTrack](https://github.com/ifzhang/ByteTrack).

| Dataset         | HOTA | IDF1 | MOTA | Model                                                        |
| --------------- | ---- | ---- | ---- | ------------------------------------------------------------ |
| DanceTrack-val  | 59.3 | 60.6 | 89.5 | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |
| DanceTrack-test | 62.2 | 63.0 | 91.6 | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |
| MOT17-half-val  | 67.1 | 78.0 | 75.8 | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |
| MOT17-test      | 63.6 | 78.7 | 79.9 | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |
| MOT20-test      | 62.5 | 78.4 | 76.7 | [Google Drive](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) |


* For more YOLO-X weights, please refer to the model zoo of [ByteTrack](https://github.com/ifzhang/ByteTrack).

### ReID Model

Ours ReID models for **MOT17/MOT20** is the same as [BoT-SORT](https://github.com/NirAharon/BOT-SORT) , you can download from [MOT17-SBS-S50](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing), [MOT20-SBS-S50](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing), ReID models for DanceTrack is trained by ourself, you can download from [DanceTrack](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing).

**Notes**:


* [MOT20-SBS-S50](https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN?usp=sharing) is trained by [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT), because the weight from BOT-SORT is corrupted. Refer to [Issue](https://github.com/GerardMaggiolino/Deep-OC-SORT/issues/6).
* ReID models for DanceTrack is trained by ourself, with both DanceTrack and CUHKSYSU datasets.

## Training

### Train the Detection Model

You can use DPTrack without training by adopting existing detectors. But we borrow the training guidelines from ByteTrack in case you want work on your own detector. 

Download the COCO-pretrained YOLOX weight [here](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0) and put it under *\<DPTrack_HOME\>/pretrained*.

* **Train ablation model (MOT17 half train and CrowdHuman)**

  ```shell
  python3 tools/train.py -f exps/example/mot/yolox_x_ablation.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
  ```

* **Train MOT17 test model (MOT17 train, CrowdHuman, Cityperson and ETHZ)**

  ```shell
  python3 tools/train.py -f exps/example/mot/yolox_x_mix_det.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
  ```

* **Train MOT20 test model (MOT20 train, CrowdHuman)**

  For MOT20, you need to uncomment some code lines to add box clipping: [[1]](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/data_augment),[[2]](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/datasets/mosaicdetection.py#L122),[[3]](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/datasets/mosaicdetection.py#L217) and [[4]](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/utils/boxes.py#L115). Then run the command:

  ```shell
  python3 tools/train.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
  ```

* **Train on DanceTrack train set**

  ```shell
  python3 tools/train.py -f exps/example/dancetrack/yolox_x.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
  ```

* **Train custom dataset**

  First, you need to prepare your dataset in COCO format. You can refer to [MOT-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_mot17_to_coco.py) or [CrowdHuman-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_crowdhuman_to_coco.py). Then, you need to create a Exp file for your dataset. You can refer to the [CrowdHuman](https://github.com/ifzhang/ByteTrack/blob/main/exps/example/mot/yolox_x_ch.py) training Exp file. Don't forget to modify get_data_loader() and get_eval_loader in your Exp file. Finally, you can train bytetrack on your dataset by running:

  ```shell
  python3 tools/train.py -f exps/example/mot/your_exp_file.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
  ```

### Train the ReID Model

After generating MOT ReID dataset as described in the 'Data Preparation' section.

```shell
cd <BoT-SORT_dir>

# For training MOT17 
python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/MOT17/sbs_S50.yml MODEL.DEVICE "cuda:0"

# For training MOT20
python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/MOT20/sbs_S50.yml MODEL.DEVICE "cuda:0"

# For training DanceTrack, we joint the CHUKSUSY to train ReID Model for DanceTrack
python3 fast_reid/tools/train_net.py --config-file ./fast_reid/configs/CUHKSYSU_DanceTrack/sbs_S50.yml MODEL.DEVICE "cuda:0"
```

Refer to [FastReID](https://github.com/JDAI-CV/fast-reid)  repository for addition explanations and options.

## Tracking

**Notes**:


* Please note that the current code has not yet been systematically organized, and the training and testing sections are still being refined. The core code can be found in the **DPTrack/trackers/dp_tracker** directory. We are working hard to optimize the code structure to improve usability and accessibility. 

Thank you for your understanding and support. If you have any questions or suggestions, please feel free to reach out.



## Acknowledgement

A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [OC-SORT](https://github.com/noahcao/OC_SORT), [ByteTrack](https://github.com/ifzhang/ByteTrack), [BoT-SORT](https://github.com/NirAharon/BOT-SORT), [FastReID](https://github.com/JDAI-CV/fast-reid). Many thanks for their wonderful works.

