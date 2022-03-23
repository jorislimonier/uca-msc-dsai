# Notes on visualizing and Understanding Convolutional Networks
## Notes on paper
### Introduction
- CNN's are great, but we don't really know why
- Visualization on Deconvnet (2011)
- Work is on supervised learning
- Goal: show which patterns from the training set activate the feature map
- Previous work had shown regions of the image that are important. This paper shows "top-down projections that reveal structures within each patch that stimulate a particular feature map"

### Approach
- Models: standard fully supervised convnet models
- 
### ...
- Ablation of each layer to understand its importance
- Deconvnet steps:
    - Unpool 
        - Max-pooling is non-invertible, but **pooling indices** (*switch* variables) help
    - Rectification
        - ReLU ensures positivness ($ReLU(x) = \max(0, x)$)
    - Filter
        - Apply transposed filters to the rectified maps (flip each filter vertically and horizontally)




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