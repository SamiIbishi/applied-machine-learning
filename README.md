# applied-ml
 Application Challenges for Machine Learning am Beispiel von IBM Power Architecture.

## Requirements

Python >= 3.8
CudaToolkit = 11.0
Pytorch = 1.7
Tensorboard = 2.4

## Conda Environment 

For this project we used a conda environment - "applied-ml". 

In order to use it you need to have Anaconda or Miniconda installed on your machine.  
Open up the terminal and navigate to the directory where the project is located. 

Execute the following command:

    $ conda env create -f environment.yml

That should create the conda enviroment "applied-ml". Finally activate it by:

	$ conda activate applied-ml

Now, you should be able to execute all scripts and notebooks. 

## Folder Structure
  ```
  applied-machine-learning/
  │
  ├── .gitignore
  ├── environment.yml
  ├── LICENSE
  ├── README.md
  │
  ├── data/ - default directory for storing input data
  │
  ├── data_loader/ - anything about data loading goes here
  │   ├── FaceRecognitionDataset.py
  │   └── data_loaders.py - NOT EXISTING YET
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── logger.py - NOT EXISTING YET
  |   └── ...
  │  
  ├── model/ - models, losses, and metrics
  │   ├── FaceNet.py
  │   ├── model.py - NOT EXISTING YET
  │   ├── metric.py - NOT EXISTING YET
  │   └── loss.py - NOT EXISTING YET
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   ├── FaceNetTrainer.py
  │   └── trainer.py - NOT EXISTING YET
  │
  └── utils/ - small utility functions
      ├── util.py - NOT EXISTING YET
      └── ...
  ```

## License
This project is licensed under the MIT License. See LICENSE for more details.