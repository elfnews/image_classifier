# Image Classifier with Python

Python based for Image Classifier using pre-trained neural network. PyTorch based neural network notebook. There is a command line version of the application. The CLI supports that training, saving of the neural network using "train.py". Loading and prediction using "predict.py"

# Training and saving of neural network
```
python train.py -h
usage: train.py [-h] [-s CHECKPOINTS_DIRECTORY] [-a MODEL_ARCHITECTURE] [-l LEARNING_RATE] [-d DROPOUT_PERCENTAGE] -i NO_INPUTS -o NO_OUTPUTS [-u LAYER_SIZES] [-e EPOCHS] [-g] [-v] data_directory

Train a machine learning model

positional arguments:
  data_directory        Set directory for data

optional arguments:
  -h, --help            show this help message and exit
  -s CHECKPOINTS_DIRECTORY, --save_dir CHECKPOINTS_DIRECTORY
                        Set directory to save checkpoints
  -a MODEL_ARCHITECTURE, --arch MODEL_ARCHITECTURE
                        Machine learning architecture to use (default: vgg16)
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        Set learning rate (default: 0.05)
  -d DROPOUT_PERCENTAGE, --dropout_percentage DROPOUT_PERCENTAGE
                        Set dropout percentage (default: 0.02)
  -i NO_INPUTS, --no_inputs NO_INPUTS
                        Set number of inputs to the model (default: 25088)
  -o NO_OUTPUTS, --no_outputs NO_OUTPUTS
                        Set number of outputs from the model (default: 102)
  -u LAYER_SIZES, --layer_size LAYER_SIZES
                        Specify layer sizes (default: [4096, 256])
  -e EPOCHS, --epochs EPOCHS
                        Set number of Training Epochs (default: 20)
  -g, --gpu             Use GPU if available for training (default: False)
  -v, --version         Displays the version of the program
```
# Loading and prediction of neural network
```
python predict.py -h
usage: predict.py [-h] [-t TOPK] [-c CATEGORY_NAMES_FILE] [-g] [-v] image_file checkpoint_file

Predict using a pre-trained machine learning model

positional arguments:
  image_file            Path to the input image file
  checkpoint_file       Path to the checkpoint file to load model

optional arguments:
  -h, --help            show this help message and exit
  -t TOPK, --top_k TOPK
                        Number of top most likely cases to return (default: 5)
  -c CATEGORY_NAMES_FILE, --category_names CATEGORY_NAMES_FILE
                        Path to JSON file that maps category names and the indices
  -g, --gpu             Use GPU if available for training (default: False)
  -v, --version         Displays the version of the program
```
