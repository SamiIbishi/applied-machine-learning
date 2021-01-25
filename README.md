# Applied Machine Learning

> This project was build in the scope of an advanced practical course "Application Challenges 
for Machine Learning utilizing IBM Power Architecture" at Technical University Munich. Which is supported via the cooperation between TU Munich and IBM.
The overall task is to build a robust face recognition model. Including the possibility of one shot learning. Additionally, ethical questions regarding 
face recognition should be elaborated.
[OpenPower@TUM](https://openpower.ucc.in.tum.de/home2/education/teaching-and-practical-courses/winter-2020-2021/)

<hr>

# Table of Contents
* [Requirements](#requirements)
* [Conda Environment](#conda-environment)
* [Face Recognition](#face-recognition)
    * [Streamlit App](#streamlit-app)
    * [Jupyter Notebooks](#jupyter-notebooks)
* [Folder Structure](#folder-structure)
* [License](#license)


## <a name="requirements"></a> Requirements

Python >= 3.8       <br />
CudaToolkit = 11.0  <br />
Pytorch = 1.7       <br />
Tensorboard = 2.4   <br />
Streamlit = 0.74.1  <br />

## <a name="conda-environment"></a> Conda Environment 

For this project we used a conda environment - "applied-ml". 

In order to use it you need to have Anaconda or Miniconda installed on your machine.  
Open up the terminal and navigate to the directory where the project is located. 

Execute the following command:

    $ conda env create -f environment.yml

That should create the conda enviroment "applied-ml". Finally, activate it by:

	$ conda activate applied-ml

Now, you should be able to execute all scripts and notebooks. If the environment yaml 
file changed since creation, you can use the following command to update your conda
environment. This command will add the new packages and delete all packages that are not
used anymore.

    $ conda env update -f environment.yml --prune

If you work on this repository and need to add new packages. Install them first while
being in the conda environment. Then use the following command:

    § conda env export | grep -v "^prefix: " > environment.yml

Be aware that this command includes linux terminal specific command ('grep') which means
that it can't be executed on a normal Windows terminal.You can use a special terminal that
interprets this commands. Alternatively, you can first update the yaml file with the following 
command and then delete the prefix entry manually. 

    $ conda env export > environment.yml

## <a name="face-recognition"></a> Face Recognition

### <a name="streamlit-app"></a> Streamlit App
We added the streamlit app to visualize the functionalities of our face recognition application. 
It includes a short description of the application, a tutorial, as well as a one shot learning
part. Use the following command in the terminal to start the streamlit app:

    $ streamlit run FaceRecognitionApp.py

### <a name="jupyter-notebooks"></a> Jupyter Notebooks 
We added several jupyter notebooks to explore the dataset, investigate the model, and to 
create a machine learning pipeline. To use one of the following commands in the terminal to
start the jupyter notebooks:

    $ jupyter lab

or 

    $ jupyter notebook

## <a name="folder-structure"></a> Folder Structure
  ```
  applied-machine-learning/
  │
  ├── .gitignore
  ├── environment.yml
  ├── LICENSE
  ├── README.md
  │
  ├── FaceRecognitionApp.py
  ├── Playground.ipynb
  ├── ToyPipeline_FaceNet.ipynb
  ├── Pipeline_FaceNet.ipynb
  │
  ├── documentation/
  │   └── images/
  │
  ├── tutorial_images/
  │   ├── anchor_images/
  │   └── test_images/
  │
  ├── pretrained_model/
  │   ├── model
  │   ├── hyper_parameter.json
  │   ├── model_parameter.json
  │   └── anchor_embedding.json
  |
  ├── test/
  │   ├── Pep8Validation.py
  │   ├── tensorboard_test.py
  │   └── trainer_class_test.py
  |
  └── src/
      ├── data/
      │
      ├── data_loader/ 
      │   ├── DatasetDownloader.py
      │   ├── FaceRecognitionDataset.py
      │   └── DataSplitter.py
      │
      ├── logger/ - module for visualization and logging
      │   └── logger.py - NOT EXISTING YET
      │  
      ├── model/ 
      │   └── FaceNet.py
      │
      ├── saved/
      │   ├── trained_models/
      │   └── log/ - default logdir for tensorboard and logging output
      │
      ├── trainer/ 
      │   └── FaceNetTrainer.py
      │
      └── utils/ 
          ├── hyperparameter_tuning.py
          ├── offline_training.py
          ├── utils.py
          ├── utils_images.py
          ├── utils_loss_functions.py
          ├── utils_optimizer.py
          ├── utils_pretrained_models.py
          └── utils_tensorboard.py
  ```

## License
This project is licensed under the MIT License. See LICENSE for more details.
