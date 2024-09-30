# Staying in the Cat-and-Mouse Game: Towards Black-box Adversarial Example Detection

<img src="Image\framework.jpg" style="zoom:80%;" />

**Figure 1**: Overview of the proposed method consisting of (a) Data reconstruction, which reconstruct the original input images; (b) Feature extraction, which extracts the layer activations of the images before and after reconstruction; and (c) Adversarial example detection, which discriminate between the normal and adversarial examples through the reconstruction errors.

## Requirements

1. [Python](https://www.python.org/)
2. [Pytorch](https://pytorch.org/)
3. [Torchvision](https://pytorch.org/vision/stable/index.html)
4. [Robustness](https://pypi.org/project/robustness/)

## Black-box Adversarial Example Detection

<img src="Image\introduction.jpg" style="zoom:20%;" />

**Figure 2.** Detection performance on white-box and black-box adversarial example. Existing adversarial example detection methods (LID, MD and SID) manifest significant performance degradation under black-box attacks. PRD and FRD are our proposed data reconstruction-based methods, which improve under both white-box and black-box attacks. SM and EM denote different black-box settings.

## Part 0. Victim and Threat Model Settings

* Test accuracy of the victim model and threat model on CIFAR-10 and CIFAR-100 dataset.
* Details are available in [**config**](checkpoints/README.txt).

|          | ResNet18(NT) | ResNet18(AT) | VGG16(NT) | VGG16(AT) | WResNet28(NT) | WResNet28(AT) |
| :-------: | :----------: | :----------: | :-------: | :-------: | :-----------: | :-----------: |
| CIFAR-10 |    94.60%    |    87.36%    |  93.42%  |     -     |    96.14%    |       -       |
| CIFAR-100 |    77.70%    |    51.75%    |  68.47%  |     -     |    77.89%    |       -       |

## Part 1. Data Reconstruction Module Training

* Train a PixelVAE or a FreqVAE as the data reconstruction module, which is the basis of adversarial detection.

```python
cd blackbox-detection
python freq-train-pipeline.py
python spat-train-pipeline.py
```

## Part 2. Adversarial Example Detection

* Generate adversarial examples.
* We perform non-target attacks to get the adversarial examples.
* Details are available in [**config**](dataset/README.txt).

|  Attack  | step | strength | attack-step |      norm      |
| :------: | :--: | :------: | :---------: | :------------: |
|   PGD   |  5  |   1.0   |     0.5     |   $l_{2}$   |
|   PGD   | 100 |  8/255  |    2/255    | $l_{\infty}$ |
|   FGSM   |  1  |   0.01   |      -      |   $l_{2}$   |
|   BIM   |  5  |   0.01   |      -      |   $l_{2}$   |
| DeepFool |  5  |   0.5   |      -      |   $l_{2}$   |
|   C&W   |  5  |   0.5   |      -      |   $l_{2}$   |

```python
cd blackbox-detection
python run-generation.py
```

* Train and evaluate a adversarial example detector:
* Details are available in [**config**](config/README.txt).

|                          detection method                          |            command            |
| :----------------------------------------------------------------: | :---------------------------: |
| [LID](https://github.com/xingjunm/lid_adversarial_subspace_detection) | run-detection.py --method lid |
|     [MD](https://github.com/pokaxpoka/deep_Mahalanobis_detector)     | run-detection.py --method md |
|                [SID](https://github.com/JinyuTian/SID)                | run-detection.py --method sid |
|                       FRD (amp, pha, joint)                       |     run-frd-detection.py     |
|                                PRD                                |     run-prd-detection.py     |

* **Parameter[freqvae]**: choice: phase, amplitude, joint
* **Parameter[version]**: choice: base, online

## References

* [**LID**] (https://github.com/xingjunm/lid_adversarial_subspace_detection)
* [**MD**] (https://github.com/pokaxpoka/deep_Mahalanobis_detector)
* [**SID**] https://github.com/JinyuTian/SID)
* [**Robustness**] (https://github.com/MadryLab/robustness)

if you have any questions, please contact with {yifeigao, zylin, yunfanyang, jtsang}@bjtu.edu.cn.

## Citation

If you find this repo useful for your research, please consider citing the paper.

```
@article{
  title={Towards Black-box Adversarial Example Detection: A Data Reconstruction based Method},
  author={Yifei Gao, Zhiyu Lin, Yunfan Yang, Jitao Sang},
  journal={},
  volume={},
  year={2023}
}
```
