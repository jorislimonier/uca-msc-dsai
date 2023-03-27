## YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

## Instructions

### Tips on writing a paper review

(Based on example review forms, from the 2020 AAAI conference) Address the following parts:

- Motivation/Relevance: why did the authors do what they did?
- Related work/Novelty: how is this situated in the research field/current state-of-the-art?
- Contribution: what is the presented contribution exactly, what does it do and how does it work?
- Support/Evaluation/Correctness: how are the claims supported: theory and/or empirical evaluation?
- Impact/Significance: how will this work change the world?
- Writing/Clarity: how easy was it to read?

For most of these categories, it is good to both highlight the authors' view and your own interpretation on it. Exemplary format:

### Summary

**Summarize the main claims/contributions of the paper in your own words.**

Trainable bag-of-freebies.

### Relevance

**Is this paper relevant to an AI audience?**

Yes

### Significance

**Are the results significant?**

Yes. Significant speed improvement and decent accuracy improvement.

### Novelty

#### SOTA
SOTA object detection methods are based on the following ideas:
1. a faster and stronger network architecture
1. a more effective feature integration method
1. a more accurate detection method
1. **a more robust loss function**
1. **a more efficient label assignment method**
1. **a more efficient training method**

YOLOv7 focuses of 4., 5. and 6. above.

#### Contribution

**Are the problems or approaches novel?**

1. Design of several trainable bag-of-freebies methods, so that real-time object detection can greatly improve the detection accuracy without increasing the inference cost
1. Find two new issues for the evolution of object detection methods:
   1. How re-parameterized module replaces original module
   1. How dynamic label assignment strategy deals with assignment to different output layers.
1. New methods for the real-time object detector that can effectively utilize parameters and computation ("Extend" and "compound scaling")
1. Speed and accuracy improvements

- E-ELAN: Extended ELAN
- Model scaling for concatenation-based models
- Trainable bag-of-freebies

### Soundness

**Is the paper technically sound?**

### Evaluation

**Are claims well-supported by theoretical analysis or experimental results?**

### Clarity

**Is the paper well-organized and clearly written?**

### Detailed Comments

**Elaborate on your assessments and provide constructive feedback.**

### Questions for the authors

**Clarification questions for authors to address during the author feedback period.**
