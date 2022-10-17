# Research Project
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Introduction](#introduction)
- [Projects details](#projects-details)
  - [Deadlines](#deadlines)
- [Paper notes](#paper-notes)
  - [Deep Learning-Based Human Pose Estimation: A Survey](#deep-learning-based-human-pose-estimation-a-survey)
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
- COCO(2D): [Info & Download](https://cocodataset.org/#home)\
  - 330 000 images
  - 200 000 labeled subjects with keypoints
  - 17 joints per person
- Human3.6M (3D) [Info](http://vision.imar.ro/human3.6m/description.php), [Download](https://deepai.org/dataset/human3-6m)

### Evaluation metrics
#### 2D HPE
- Percentage of Correct Parts (PCP):
$$
\text{error in localization of limb} < k \times \text{limb size}, \qquad k \in [0.1, 0.5]
$$
It over-penalizes small limbs.

- Percentage of Correct Keypoints (PCK):
$$
\text{error in localization of limb} < k \times \text{torso size}, \qquad k \in [0.1, 0.5]
$$

#### 3D HPE
- Mean Per Joint Position Error (MPJPE): 3D Euclidian distance between truth and predictions
$$
MPJPE = \frac{1}{n} \sum_{i=1}^n \| J_i - J_i^*\|_2
$$
- PMPJPE: reconstruction error
- NMPJPE: MPJPE + normalization
- MPBE: Euclidian distance between vertices
- 3DPCK: 3D extension of PCK



## Paper notes
### Deep Learning-Based Human Pose Estimation: A Survey
#### Acronyms
- Meta analysis of >250 papers since 2014
- 2D HPE is easy, 3D HPE is hard
- 3 types of model:
  - Kinematic model: only joints
  - Planar model: Joints + rectangles to approximate human body
  - Volumetric model: Fit to skin




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

