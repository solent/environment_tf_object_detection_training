# environment_tf_object_detection_training

Environment used to train models using TensorFlow [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

## About this project

The idea is to help new users to have a turnkey environment on Ubuntu to use TensorFlow GPU and train object detection's models.

## Prerequisites

In order to use this environment, you would still have to install [tensorflow-gpu](https://www.tensorflow.org/install/install_linux) and all its dependencies.
I build this environment with TensorFlow 1.8 and I advise you to do the same.
And be careful, to install CUDA Toolkit 9.0 and cuDNN SDK v7, you have to follow the full corresponding Nvidia documentation (it takes me approximatically one day).

- Ubuntu 18.04
- Python 3.6.5
- TensorFlow GPU 1.8
- A clone of [TensorFlow models](https://github.com/tensorflow/models)

You will probably need a few more things when executing some scripts, and pip3 will be your friend for this ;)

## Structure

The project has a classic structure :

- bin: contains the executable files
- conf: contains the configuration of your environment
- data: will contains the generated .record files
- graph: will contains the generated TensorFlow graph
- images: will contains the temporary duplicated images used to train and test your model
- lib: contains the libraries used to generate the models or to do some image augmentations
- models: contains the pre-trained models
- training: contains your training configuration and will contains the generated checkpoints

## Setup

Clone this project whatever you want on your Ubuntu machine (with the completed Prerequisites).
Read the config.cfg.defaults's files and its comments (and also to have examples).
You have to duplicate some properties into the config.cfg files to configure your own environement : 
- OBJECT_DETECTION_PATH : Path of the object detection folder (into TensorFlow models installed with the Prerequisites)
- IMAGES_SOURCE_PATH: Path of your images's folder.
- DEV_PATH: Root of your workspace
- PROJECT_FOLDER: Name of your project's folder (must be into "DEV_PATH")
- ENTITIES: Array of your classes names (must also correspond to the different images folders)
- CONFIG_FILE: Model's configuration

Execute "chmod +x" on the differents bin files.

## Train your model

### If you keep the mobilenet v1 configuration

Once the setup completed, you can build a graph following this procedure :

- Update CLASSES parameter (into config.cfg) with your classes names (separated by a space like in the example)
- For each class, copy the corresponding annotated images into a folder named with the class (into the IMAGES_SOURCE_PATH folder)
- Update the training/object_detection.pbtxt file with your own classes
- Update "model.ssd.num_classes" into your CONFIG_FILE : it must correspond to your own number of classes
- Go into the bin folder
- Execute copy_images.sh: it will concatenate your images into one folder, then copy them into images/train, and finally move 10% of them into images/test
- Execute recompute_tfrecords: it will generate your .records files.
- Execute train_graph.sh: it will train your model, from step to step. You can stop it by executing "sudo kill <PID>" (and find <PID> by using nvidia-smi for example)
- Execute export_graph.sh: it will export your model from the last checkpoint.


### If you want to use another model

The procedure is the same but you have to change a few things in advance:
- Copy the pre-trained model's folder into models
- Copy the corresponding config file into training
- Update CONFIG_FILE parameter to match the new config file
- Update new config file following properties:

train_config.fine_tune_checkpoint: "../models/<YOUR_ANOTHER_PRETRAINED_MODEL_FOLDER>/model.ckpt"
train_input_reader.tf_record_input_reader.input_path: "../data/train.record"
train_input_reader.label_map_path: "../training/object-detection.pbtxt"
eval_input_reader.tf_record_input_reader.input_path: "../data/test.record"
eval_input_readerlabel_map_path: "../training/object-detection.pbtxt"


