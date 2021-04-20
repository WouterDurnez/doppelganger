#    ___                     _  ___   _   _  _
#   |   \ ___ _ __ _ __  ___| |/ __| /_\ | \| |__ _ ___ _ _
#   | |) / _ \ '_ \ '_ \/ -_) | (_ |/ _ \| .` / _` / -_) '_|
#   |___/\___/ .__/ .__/\___|_|\___/_/ \_\_|\_\__, \___|_|
#            |_|  |_|                         |___/

"""
DATA PREPARATION
-- Coded by Wouter Durnez
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mtcnn import MTCNN
from tqdm import tqdm
from helper import log, hi
from torchvision.datasets import cifar


def read_images(dir: str, max_images=None, extension='jpeg') -> np.ndarray:
    """
    Read a number of images from a directory
    :param dir: directory containing image files
    :param max_images: maximum number of images to read
    :param extension: extension to look for in folder
    :return: array of images
    """

    # Generate all image paths to jpeg
    image_paths = [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith(extension)]

    # Only read *max_images*, unless there's not even that many
    end = max_images if (max_images and max_images < len(image_paths)) else -1

    # Store images here
    images = []
    for image_path in tqdm(image_paths[:end], desc='Reading images'):
        # Read image and convert to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images.append(image)

    return np.array(images)


def save_images(images: np.ndarray, dir: str, extension='jpg'):
    """
    Save images to disk
    :param images: images to save
    :param dir: directory to save them in
    :param extension: format of choice (default jpeg)
    """
    # Create dir if it doesn't exist
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Write out files
    for idx, image in tqdm(enumerate(images), desc='Saving face(s)'):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dir, f'face{idx}.{extension}'), image)
        except Exception as e:
            log(f'Failed to save index {idx} - {e}')


def extract_faces(images: np.ndarray, buffer:float =.15, show_plot:bool=True) -> (np.ndarray, np.ndarray):
    """
    Extract faces and bounding boxes from images
    :param images: images to find faces in
    :param buffer: percentage of max(width,height) to add as a buffer around the detection
    :param show_plot: show plot or not
    :return: list of faces, list of bounding box coordinates
    """

    # Create the detector, using default weights
    detector = MTCNN()

    # Store faces and bounding boxes
    cropped_faces, boxes = [], []

    # Go over all images
    for image in tqdm(images, desc='Extracting faces'):

        # Detect faces
        faces = detector.detect_faces(image)

        # Display faces on the original image and crop them out
        if show_plot:
            plt.imshow(image)

        # Loop over all faces
        for face in faces:

            # Get coordinates and draw rectangle around face
            x, y, width, height = face['box']

            diff = np.abs(width - height) / 2

            # I want square cutouts that are slightly bigger
            if width < height:
                x = int(x - diff)
                width = height
            else:
                y = int(y - diff)
                height = width

            # Edge buffer
            x = int(x - buffer*width)
            y = int(y - buffer*width)
            width = int((1 + 2*buffer)*width)
            height = int((1 + 2*buffer)*height)

            # Crop face out of image
            image_cropped = image[y:y + height, x:x + width]

            if show_plot:
                # Get axes
                ax = plt.gca()

                # Draw box
                rect = Rectangle((x, y), width, height, fill=False, color='red')
                ax.add_patch(rect)

            cropped_faces.append(image_cropped)
            boxes.append((x, y, width, height))

        plt.show()

    return np.array(cropped_faces), np.array(boxes)


def scale_images(images: np.ndarray, dimensions: tuple) -> np.ndarray:
    """
    Scale images
    :param images: source images
    :param dimensions: new dimensions
    :return: scaled images
    """

    # Store scaled images in list
    new_images = []

    # Scale and append
    for image in tqdm(images, desc='Scaling images'):
        try:
            scaled = cv2.resize(image, dsize=dimensions, interpolation=cv2.INTER_CUBIC)
            new_images.append(scaled)
        except Exception as e:
            print('Failed to resize image:', e)

    return np.array(new_images)


if __name__ == '__main__':

    hi(title='Data prep', params={'timestamped':True})

    # Location of raw data
    raw_dir = '../data/me/raw'
    fake_dir = '../data/fake/original'

    # Get images
    faces = read_images(dir=fake_dir,extension='jpg')

    # Get faces
    #faces, boxes = extract_faces(images=images, show_plot=False)

    # Rescale faces --> Nope, let pytorch take care of that
    faces = scale_images(images=faces, dimensions=(128,128))

    # Write faces to folder
    save_images(images=faces, dir='../data/fake/rescaled')

    # Let's also download Cifar
    '''cifar_dir = '../data/cifar'
    cifar.CIFAR10(root=cifar_dir,train=True,download=True)'''

    plt.imshow(faces[0])
    plt.show()
