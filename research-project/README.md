# Research Project

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Introduction](#introduction)
- [Projects details](#projects-details)
  - [Deadlines](#deadlines)
  - [Datasets](#datasets)
    - [2D](#2d)
    - [3D](#3d)
  - [Evaluation metrics](#evaluation-metrics)
    - [2D HPE](#2d-hpe)
    - [3D HPE](#3d-hpe)
- [Paper notes](#paper-notes)
  - [01 - Deep Learning-Based Human Pose Estimation: A Survey](#01---deep-learning-based-human-pose-estimation-a-survey)
  - [02 - Evaluation of deep pose detectors for automatic analysis of film style](#02---evaluation-of-deep-pose-detectors-for-automatic-analysis-of-film-style)
- [References](#references)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction

- The project is an opportunity to develop organizational skills and autonomy.
- The subjects of research projects are deliberately complicated, and require, on the one hand, the
  mobilization of the skills acquired during the Master cycle and, on the other hand, the search for
  new knowledge.
- The work carried out must be able to position itself in relation to existing work, whether in
  relation to a scientific state of the art or to a comparative study of similar tools/solutions.

You will be evaluated on the basis of a scientific article you have written. This article must have at least 2 parts:

- a state of the art i.e. a synthesis of 2 to 4 papers selected by the proposer
- the highlighting of a research project based on the previous state of the art i.e. the definition of the first experiments to fill a gap in the state of the art or to use the state of the art to solve a new problem.

You are expected to be expressive and creative. One day per week is dedicated to this research project, as well as two weeks full-time.
Only the final report and the defense are evaluated.

## Projects details

**Title**: 3D Human Pose Estimation in Videos

**Contact**: Frédéric Precioso, frederic.precioso@univ-cotedazur.fr

**Description**:

The estimation of human poses, i.e., of every joint of every visible human, in a 3D space inferred from a 2D monocular videos is a challenging task and an intense area of research in computer vision. The challenges are number of humans, poses, occlusions, camera pose and motion, defocus, blur and partial body parts [1,2]. This project consists in

(1) studying major recent results in this domain, implementing and testing the frameworks of [3,4,5,6] on the original datasets, and

(2) if time permits, on a new film dataset [2] on which artistic shots make the task challenging.

This TER lies in the framework of the ANR TRACTIVE project (https://www.i3s.univ-cotedazur.fr/TRACTIVE/).

### Deadlines

- State of the Art (5-8 pages): November 30
- Full project (8-12 pages): February 15
- Oral presentation: Last week of February

### Datasets

<table>
  <tr>
    <th>
      Paper
    </th>
    <th>
      Dataset
    </th>
    <th>
      Algorithm
    </th>
  </tr>
  <tr>
    <td align="center">01</td>
    <td>COCO, Human3.6m</td>
    <td></td>
  </tr>
  <tr>
    <td align="center">02</td>
    <td>COCO</td>
    <td>
      HR-Net (top-down), <br>
      OpenPose (bottom-up)
    </td>
  </tr>
  <tr>
    <td align="center">03</td>
    <td></td>
    <td></td>
  </tr>
</table>

#### 2D

- [COCO](https://cocodataset.org/#home)
  - 330 000 images
  - 200 000 labeled subjects with keypoints
  - 17 joints per person
- [Max Planck Institute for Informatics (MPII)](http://human-pose.mpi-inf.mpg.de)
  - 25 000 images
  - 40 000 individuals with annotated body joints
  - 410 human activities
- [Leeds Sports Pose Extended (LSP)](https://dbcollection.readthedocs.io/en/latest/datasets/leeds_sports_pose_extended.html)

  - 10 000 images
  - From Flickr searches "parkour", "gymnastics", "athletics" (challenging poses)
  - 14 joints per image annotated using AMT

- [FLIC](https://bensapp.github.io/flic-dataset.html)
  - 5003 image dataset from popular Hollywood movies
  - 20 000 people annotated with AMT
  - Downloadable [with TF](https://www.tensorflow.org/datasets/catalog/flic)
- [AIC-HKD](https://github.com/AIChallenger/AI_Challenger_2017)
  - Probably not usable, authors only seem to provide Baidu link through GitHub issues. The links expire quickly.
- [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)
  - 20 000 images
  - 80 000 human poses with 14 labeled keypoints
  - 8 000 images in test set
- [Penn Action](https://dreamdragon.github.io/PennAction/)
  - 2326 video sequences
  - 15 different actions
  - human joint annotations for each sequence
- [J-HMDB](http://jhmdb.is.tue.mpg.de/)
  - 960 video sequences
  - 21 different actions
  - Subset of HMDB51
- PoseTrack 2
  - Website down
  - May be downloadable through [MMPose](https://github.com/open-mmlab/mmpose/blob/master/docs/en/tasks/2d_body_keypoint.md#posetrack18)
  - 66 374 frames within 514 videos
    - 300 for training
    - 50 for validation
    - 208 for testing

#### 3D

- [Human3.6M](http://vision.imar.ro/human3.6m/description.php)
  - 3.6M images
  - 11 professional actors (6 male, 5 female)
  - 17 kind of actions
- [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/)
  - 1.3M images from 14 cameras
  - 8 actions
  - 8 subjects
- [MuPoTS-3D](https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/)
  - 8 000 images
  - 8 subjects
- [HumanEva](https://github.com/openai/human-eval)
  - 40 000 images
  - 7 actions
  - 6 subjects
- [CMU Panoptic Dataset](http://domedb.perception.cs.cmu.edu/)
  - 1.5M images
  - 8 subjects
  - social actions
- [TotalCapture](https://cvssp.org/data/totalcapture/)
  - 1.9M images
  - 5 actions
  - 5 subjects
- [MuCo-3DHP Dataset](https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/)
  - ? images
  - ? actions
  - ? subjects
- [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)
  - 51 000 images
  - 7 subjects
- [AMASS](https://amass.is.tue.mpg.de/)
  - 9M images
  - 300 subjects
- [NBA2K](https://grail.cs.washington.edu/projects/nba_players/)
  - 27 000 images
  - 27 subjects
- [GTA-IM](https://zhec.github.io/hmp/)
  - 1M images
- [Occlusion-Person](https://github.com/zhezh/occlusion_person)
  - 73 000 images
  - 20% occluded


### Evaluation metrics

#### 2D HPE

- Object-Keypoint Similarity (OKS)
  $$OKS = \frac{\sum_{i=1}^{n} \exp \left(-d_i^2 / 2s^2 \kappa_i^2\right) \delta_{v_i > 0}}{\sum_{i=1}^{n} \delta_{v_i > 0}}$$
  where:
  - $s$: object scale
  - $d_i$: Euclidian distance of predicted joint $i$ from ground truth
  - $\kappa_i$ Per-keypoint constant that controls fallof
  - $v_i$: Visibility flag, equals
    - 0 if not labeled
    - 1 if labeled but not visible
    - 2 if labeled and visible
- Average Precision (AP)
  $$AP = \frac{TP}{PP}$$

- Average Recall (AR)
  $$AP = \frac{TP}{P}$$

- Percentage of Correct Parts (PCP)
  $$\text{error in localization of limb} < k \times \text{limb size}, \qquad k \in [0.1, 0.5]$$

  It over-penalizes small limbs.

- Percentage of Correct Keypoints (PCK)
  $$\text{error in localization of limb} < k \times \text{torso size}, \qquad k \in [0.1, 0.5]$$

#### 3D HPE

- Mean Per Joint Position Error (MPJPE): 3D Euclidian distance between truth and predictions
  $$MPJPE = \frac{1}{n} \sum_{i=1}^n \| J_i - J_i^*\|_2$$
- PMPJPE: reconstruction error
- NMPJPE: MPJPE + normalization
- MPBE: Euclidian distance between vertices
- 3DPCK: 3D extension of PCK

## Paper notes

### 01 - Deep Learning-Based Human Pose Estimation: A Survey

- Meta analysis of >250 papers since 2014
- 2D HPE is easy, 3D HPE is hard
- 3 types of model:
  - Kinematic model: only joints
  - Planar model: Joints + rectangles to approximate human body
  - Volumetric model: Fit to skin

### 02 - Evaluation of deep pose detectors for automatic analysis of film style

Topics:

- Quantification of on-screen character poses, potentially through HPE
- Review of 1 non-film dataset + 2 film datasets (>2700 images in total)
- Identify cases where current HPE methods fail on non-film data

## References

- [1] C. Zheng et al., “Deep Learning-Based Human Pose Estimation: A Survey,” arXiv:2012.13392 [cs], Jan. 2021, Accessed: Jan. 12, 2022. [Online]. Available: http://arxiv.org/abs/2012.13392
- [2] H.-Y. Wu, L. Nguyen, Y. Tabei, and L. Sassatelli, “Evaluation of deep pose detectors for automatic analysis of film style,” in EUROGRAPHICS Workshop on Intelligent Cinematography and Editing, Reims, France, 2022, p. 9.
- [3] W. Li, H. Liu, H. Tang, P. Wang, and L. V. Gool, “MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation,” in 2022 IEEE/CVF International Conference on Computer Vision (ICCV), Montreal, QC, Canada, Jun. 2022.
- [4] Z. Liu et al., “Deep Dual Consecutive Network for Human Pose Estimation,” in 2021 IEEE/CVF [1] Conference on Computer Vision and Pattern Recognition (CVPR), Nashville, TN, USA, Jun. 2021, pp. 525–534. doi: 10.1109/CVPR46437.2021.00059.
- [5] W.-L. Wei, J.-C. Lin, T.-L. Liu, and H.-Y. M. Liao, “Capturing Humans in Motion: Temporal-Attentive 3D Human Pose and Shape Estimation From Monocular Video,” in 2022 IEEE/CVF International Conference on Computer Vision (ICCV), Montreal, QC, Canada, Jun. 2022.
- [6] Z. Liu et al., “Temporal Feature Alignment and Mutual Information Maximization for Video-Based Human Pose Estimation,” in 2022 IEEE/CVF International Conference on Computer Vision (ICCV), Montreal, QC, Canada, Jun. 2022.
- [7] Z. Qiu, Q. Yang, J. Wang, and D. Fu, “IVT: An End-to-End Instance-guided Video Transformer for 3D Pose Estimation.” in ACM Internation Cnference on Multimedia, Oct. 2022. [Online]. Available: http://arxiv.org/abs/2208.03431
- [8] Z. Li, B. Xu, H. Huang, C. Lu, and Y. Guo, “Deep Two-Stream Video Inference for Human Body Pose and Shape Estimation,” in 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Waikoloa, HI, USA, Jan. 2022, pp. 637–646. doi: 10.1109/WACV51458.2022.00071.
- [9] J. Zhang, Z. Tu, J. Yang, Y. Chen, and J. Yuan, “MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video,” in 2022 IEEE/CVF International Conference on Computer Vision (ICCV), Montreal, QC, Canada, Jun. 2022.
