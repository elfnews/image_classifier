import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

from network import Network
from train import initializeDevice

device = None
cat_to_name = []


def initializeCatgories(category_names_file):
    print(f"Loading category names from [{category_names_file}].")
    with open(category_names_file, 'r') as f:
        return json.load(f)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # Resize & Crop the image
    image.thumbnail((256, 256))
    image_resized_cropped = image.crop((0, 0, 224, 224))

    # Normalize the image use mean and standard deviation
    np_image = np.array(image_resized_cropped).astype(np.float)
    np_image = np_image / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Switch color from PIL third dimension to PyTorch first dimension
    np_image = np_image.transpose(2, 0, 1)

    return np_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    image_pil = Image.open(image_path)
    input = torch.from_numpy(process_image(image_pil))
    input = input.unsqueeze(0)
    model = model.double()
    model.eval()
    logps = model.forward(input)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    model.train()
    return top_p, top_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict using a pre-trained machine learning model")

    default_topk = 5
    default_isGPU = False

    parser.add_argument('image_file', action="store", help="Path to the input image file")
    parser.add_argument('checkpoint_file', action="store", help="Path to the checkpoint file to load model")
    parser.add_argument('-t', '--top_k', action="store", default=default_topk, dest="topk", type=int,
                        help=f"Number of top most likely cases to return (default: {default_topk})")
    parser.add_argument('-c', '--category_names', action="store", dest="category_names_file",
                        help="Path to JSON file that maps category names and the indices")
    parser.add_argument('-g', '--gpu', action="store_true", default=default_isGPU, dest="isGPU",
                        help=f"Use GPU if available for training (default: {default_isGPU})")
    parser.add_argument('-v', '--version', action="version", version='%(prog)s 1.0',
                        help="Displays the version of the program")

    results = parser.parse_args()
    print(f"Predicting using the following parameters :{results}")

    invalid_args = False
    if not os.path.isfile(results.image_file):
        print(f"Could not find image file[{results.image_file}].")
        invalid_args = True

    if not os.path.isfile(results.checkpoint_file):
        print(f"Could not find check point file[{results.checkpoint_file}].")
        invalid_args = True

    if results.category_names_file and not os.path.isfile(results.category_names_file):
        print(f"Could not find category names file[{results.category_names_file}].")
        invalid_args = True

    if invalid_args:
        sys.exit(-1)

    device = initializeDevice(results.isGPU)

    if results.category_names_file:
        cat_to_name = initializeCatgories(results.category_names_file)

    model = Network.load_checkpoint(results.checkpoint_file)

    p, c = predict(image_path=results.image_file, model=model, topk=results.topk)
    print(f"P{p} C{c}")

    labels = []
    if cat_to_name and len(cat_to_name) > 0:
        for aclass in c[0].numpy():
            labelkeypos = list(model.train_data_class_to_idx.values()).index(aclass)
            labelkey = list(model.train_data_class_to_idx.keys())[labelkeypos]
            labels.append(cat_to_name[labelkey])

    if len(labels) > 0:
        print(f"The image is one of the following flower (in decreasing likelihood):{labels}")
