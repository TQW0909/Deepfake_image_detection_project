# Deepfaked Image Detection Project

A brief introduction to the machine learning project, its objectives, and the problem it aims to solve.

## Table of Contents

<!-- - [Installation](#installation)
- [Project Motivation](#project-motivation)
- [File Descriptions](#file-descriptions)
- [How to Run](#how-to-run)
  - [Data Processing](#data-processing)
  - [Training the Model](#training-the-model)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)

## Installation

Details about the environment, libraries, and tools required to run the project. Provide commands if necessary. -->


## Datasets

# Deepfake and real images (Training and Testing)

256 X 256 jpg images

Already partitioned into train, dev, and test

No 'metadata.csv'

```
deepfake_and_real_images/
│
├── Test/  
│   ├── Fake    # 5492 images
│   ├── Real    # 5413 images
|
├── Train/ 
│   ├── Fake    # 70.0k images
│   ├── Real    # 70.0k  images
|
├── Validation/ 
│   ├── Fake     # 19.6k  images
│   ├── Real     # 19.8k  images
```

# Deepfake_faces (Testing only)

224 X 224 jpg images

Not partitioned into train, dev, and test

`metadata.csv' provides label and details for each image

```
deepfake_faces/ # Total: 95634
│
├── faces_244/  
│   ├── FAKE    # 83%
│   ├── Real    # 17%
├── metadata.csv
|
```