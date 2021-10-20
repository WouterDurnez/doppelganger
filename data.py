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
from glob import glob
from typing import Iterable, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mtcnn import MTCNN
from tqdm import tqdm

from helper import log, hi


def get_image_paths(dir: str, extensions: Union[str, Iterable] = 'jpg') -> list:
    """
    Get paths to images with specific extensions
    :param dir: directory in which to search
    :param extensions: valid image extensions
    :return: list of file paths
    """

    # Extensions must be string or Iterable
    assert isinstance(extensions, str) or isinstance(extensions,
                                                     Iterable), 'Extension must be a string or an iterable of strings!'

    if type(extensions) == str: extension = [extensions]

    # Collect all image paths
    image_paths = []
    for ext in extensions:
        image_paths += glob(f'{dir}/*.{ext}')

    return image_paths


def read_image(file_path: str) -> np.ndarray:
    """
    Read an image from a file path
    :param file_path: path to image
    :return: image in ndarray format
    """

    # Read image and convert to RGB
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(image: np.ndarray, file_name:str, dir: str, extension='jpg'):
    """
    Save image to disk
    :param image: image to save
    :param dir: directory to save it to
    :param extension: format of choice (default jpg)
    """
    # Create dir if it doesn't exist
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Write file
    try:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dir, f'{file_name}.{extension}'), image)
    except Exception as e:
        log(f'Failed to save image! - {e}',color='red')


def extract_faces(image: np.ndarray, buffer: float = .15, show_plot: bool = True)\
        -> (list,list):
    """
    Extract faces and bounding boxes from images
    :param image: picture to find faces in
    :param buffer: percentage of max(width,height) to add as a buffer around the detection
    :param square: whether we want square faces (haha)
    :param show_plot: show plot or not
    :return: list of faces, list of bounding box coordinates
    """

    # Create the detector, using default weights
    detector = MTCNN()

    # Store faces and bounding boxes
    cropped_faces, boxes = [], []

    # Detect faces
    faces = detector.detect_faces(image)

    # Display faces on the original image and crop them out
    if show_plot:
        plt.imshow(image)
        plt.axis('off')

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
        x = int(x - buffer * width)
        y = int(y - buffer * width)
        width = int((1 + 2 * buffer) * width)
        height = int((1 + 2 * buffer) * height)

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

    return cropped_faces, boxes


def scale_image(image: np.ndarray, dimensions: tuple) -> np.ndarray:
    """
    Scale images
    :param image: source image
    :param dimensions: new dimensions
    :return: scaled images
    """

    # Scale and append
    try:
        scaled = cv2.resize(image, dsize=dimensions, interpolation=cv2.INTER_CUBIC)
        return scaled
    except Exception as e:
        print('Failed to resize image:', e)


if __name__ == '__main__':
    hi(title='Data prep', timestamped = True)

    # Directories
    source_dir = '../data/me/raw'
    save_dir = '../data/me/faces'

    # Parameters
    dim = 256

    # Get images
    image_paths = get_image_paths(dir=source_dir, extensions=['jpg', 'jpeg'])

    # Process images
    for idx, image_path in tqdm(enumerate(image_paths), 'Processing images'):

        # Read images
        image = read_image(image_path)

        # Get faces
        faces, boxes = extract_faces(image=image, show_plot=True, buffer=.4)

        # Rescale face
        for face in faces:
            face = scale_image(image=face, dimensions=(dim,dim))

            # Write faces to folder
            save_image(image=face, file_name=f'face{dim}x{dim}_{idx}', dir=save_dir)

