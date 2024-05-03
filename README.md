# Deepfaked Image Detection Project

The following repo includes all the files related to the deepfake image detection project. 

## Installation

All the required libraries and packages are stored in the `requirements.txt'.

Install using:

 ```pip install -r requirements.txt```

 For downloading the datasets that are being used, visit the Kaggle URL for each dataset.


## Datasets

### Deepfake and real images (Training and Testing)

Dataset Url: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images?resource=download

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

### Deepfake_faces (Testing only)

Dataset Url: https://www.kaggle.com/datasets/dagnelies/deepfake-faces/data

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

## To Run

### Dataset Subset Generation

Using the script `dataset_maker.py` located in the `Utils` directory, a subset with a stated number of samples in each of train, validation, and test set can be generated. Note that this script is used for the  `Deepfake and real images` dataset.

Also note that the location of the orginal dataset, location of produced subset and number of samples in each set need to be changed in the script.

To run:
```
python3 dataset_maker.py
```

Using the script `testdata_maker.py` located in the `Utils` directory, a subset with a stated number of samples in test set can be generated. Note that this script is used for the  `Deepfake_faces` dataset.

Also note that the location of the orginal dataset, location of produced subset, and the number of samples in test set need to be changed in the script.

To run:
```
python3 testdata_maker.py
```

### Training and testing

Before training and testing, ensure that the paths to the training, validation, and testing datasets are updated in `Utils/dataloader.py`.

The training and testing components are all contained in `train_test.py`, and to run use the following:

```
python3 train_test.py <required arguments> <optional arguemnts>
```

With the following arguments:

Required:

- model: 'baseline_cnn' or 'cnn'
- testset: 'train' or 'test' or 'both'
  
Optional:

- --test: If we are going to test the trained model on the testsets stated above
- --tune: If we are going to doing hyperparameter tuning (no testing will be done regardless of --test flag)
- --plot: If we want to have the train and dev accuracy and loss plots in a gaph
- --GPU: If we want to utilize GPU (checks for all available GPUs and utilizes all)

#### Running on CARC

Please use the following to set up running on CARC:

```
module load conda
mamba init bash
source ~/.bashrc
mamba create --name project
mamba activate project

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install all the required packages and libaries as well
```

Then, there are two choices to run the program: as a job or in the command line.

For a job we need to use the `run.sl` file and modify to run the desired processes. To submit the job we use:

``` 
sbatch run.sl
```

For running on the command line, do the following:

```
salloc --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --time=3:00:00 --mem=32GB --gres=gpu:2

python3 train_test.py <required arguments> <optional arguemnts>
```