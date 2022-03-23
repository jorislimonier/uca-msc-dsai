# Notes on visualizing and Understanding Convolutional Networks
## Notes on paper
### 1. Introduction
- CNN's are great, but we don't really know why
- Visualization on Deconvnet (2011)
- Work is on supervised learning
- Goal: show which patterns from the training set activate the feature map
- Previous work had shown regions of the image that are important. This paper shows "top-down projections that reveal structures within each patch that stimulate a particular feature map"

### 2. Approach
- Models: standard fully supervised convnet models (AlexNet and LeCun's original paper on Deep Learning).
- Ablation of each layer to understand its importance
- Deconvnet steps:
    - Unpool 
        - Max-pooling is non-invertible, but **pooling indices** (*switch* variables) help
    - Rectification
        - ReLU ensures positivness ($ReLU(x) = \max(0, x)$)
    - Filter
        - Apply transposed filters to the rectified maps (flip each filter vertically and horizontally)

### 3. Training details
Nothing primordial, only details on parameter values.

### 4. Convnet visualization
- Fixing some issues about Krizhevsky et al.'s paper.
    - Problem 1: First layer contain high and low frequency information, but nothing in-between
    - Problem 2: Second layer contains aliasing artifacts caused by large stride (stride of 4)
    - Solution 1: Reduce filter of layer 1 from $11 \times 11$ to $7 \times 7$
    - Solution 2: Make slide 2 rather than 4
    - Fig 6 shows improvements
- Occlusion sensitivity:
    - Change probability prediction depending on which region of the image is occluded.
- Correspondance analysis:
    > Deep models differ from many existing recognition approaches in that there is no explicit mechanism for establishing correspondence between specific object parts in different images (e.g. faces have a particular spatial configuration of the eyes and nose). However, an intriguing possibility is that deep models might be implicitly computing them.
    - More recent models may have looked into that more deeply?

### 5. Experiments
- > both our feature representation and the hand-crafted features are designed using images beyond the Caltech and PASCAL training sets.
#### Varying ImageNet Model Sizes
- Removing fully connected layers (6, 7) $\implies$ little increase in error
- Removing two of the middle convolutional layers $\implies$ little increase in error
- Removing fully connected layers (6, 7) and two of the middle convolutional layers $\implies$ dramatical error increase
- Doubling/Halving the size of FC (fully connected) layers $\implies$ little difference
**$\implies$ depth needed**
- Increase size of middle convolution layer $\implies$ gain in performance
- Increase size of middle convolution layer and increase size of FC layers $\implies$ over-fitting

#### Generalization
- Good generalization (Caltech-101, Caltech-256)

## Remarks
- Easy to read, beginner friendly -> recall of some very basic concepts to talk to a large audience (explanation of $N$ samples, what $\{x, y\}$ are...etc)

## Instructions for exam
- Understand the paper
- Explain to others
- Which problem does it solve?
- What solution is proposed?
- Results?
- Opinion on the quality of the papers (*e.g.* small dataset, compared to poor-performing algorithms)
- Opinion on the paper (why it's clever, why it's a good idea)
- Ask questions during other people's presentation (min 2 questions)
- If adding resources from the paper, mention the source