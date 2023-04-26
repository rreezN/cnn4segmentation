# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from PIL import Image
from glob import glob
import numpy as np
from tqdm import tqdm
from torchvision.transforms import CenterCrop


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    train_images = []
    train_labels = []
    for i in range(24):
        train_images += glob(f"{input_filepath}train_images/train_{i}_*")
        train_labels += glob(f"{input_filepath}train_labels/labels_{i}_*")

    #                   TRAIN
    train_data = []
    train_labels_arr = []
    iterator = tqdm(zip(train_images, train_labels), total=len(train_images))
    for train_image, train_label in iterator:
        img = Image.open(train_image)
        label = Image.open(train_label)
        label = CenterCrop(92)(label)
        train_data.append(np.array(img))
        train_labels_arr.append(np.array(label))

    np.save(f"{output_filepath}/training_data.npy", np.array(train_data))
    np.save(f"{output_filepath}/training_labels.npy", np.array(train_labels_arr))

    #                   VALIDATION
    val_images = []
    val_labels = []
    for i in range(24, 27):
        val_images += glob(f"{input_filepath}/train_images/train_{i}_*")
        val_labels += glob(f"{input_filepath}/train_labels/labels_{i}_*")

    val_data = []
    val_labels_arr = []
    iterator = tqdm(zip(val_images, val_labels), total=len(val_images))
    for val_image, val_label in iterator:
        img = Image.open(val_image)
        label = Image.open(val_label)
        label = CenterCrop(92)(label)
        val_data.append(np.array(img))
        val_labels_arr.append(np.array(label))

    np.save(f"{output_filepath}/val_data.npy", np.array(val_data))
    np.save(f"{output_filepath}/val_labels.npy", np.array(val_labels_arr))

    #                   TEST
    test_images = []
    test_labels = []
    for i in range(27, 30):
        test_images += glob(f"{input_filepath}/train_images/train_{i}_*")
        test_labels += glob(f"{input_filepath}/train_labels/labels_{i}_*")

    test_data = []
    test_labels_arr = []
    iterator = tqdm(zip(test_images, test_labels), total=len(test_images))
    for test_image, test_label in iterator:
        img = Image.open(test_image)
        label = Image.open(test_label)
        label = CenterCrop(92)(label)
        test_data.append(np.array(img))
        test_labels_arr.append(np.array(label))

    np.save(f"{output_filepath}/test_data.npy", np.array(test_data))
    np.save(f"{output_filepath}/test_labels.npy", np.array(test_labels_arr))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()