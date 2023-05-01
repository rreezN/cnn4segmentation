# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from PIL import Image
import random
from glob import glob
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from albumentations import ElasticTransform, Compose


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around its centre point
    """

    image_size = (image.size[1], image.size[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image.crop((x1, y1, x2, y2))


def augment_image(img, label):

    if random.random() < 0:
        # Rotate image
        rotate_degrees = random.randint(0, 359)
        rotated_img = img.rotate(rotate_degrees, expand=True)
        rotated_label = label.rotate(rotate_degrees, expand=True)

        # Rescale image
        image_height, image_width = img.size[0:2]
        w, h = largest_rotated_rect(image_height, image_width, math.radians(rotate_degrees))

        augmented_img = crop_around_center(rotated_img, w, h)
        augmented_label = crop_around_center(rotated_label, w, h)

        augmented_img = augmented_img.resize(img.size)
        augmented_label = augmented_label.resize(img.size)
    else:
        crop_sizing = random.randint(50, 80) / 100
        # Crop image
        crop_size = (int(img.size[0] * crop_sizing), int(img.size[1] * crop_sizing))
        top_left_corner = (random.randint(0, img.size[0] - crop_size[0]),
                           random.randint(0, img.size[1] - crop_size[1]))
        cropped_img = img.crop((top_left_corner[0], top_left_corner[1],
                                         top_left_corner[0] + crop_size[0],
                                         top_left_corner[1] + crop_size[1]))
        cropped_label = label.crop((top_left_corner[0], top_left_corner[1],
                                top_left_corner[0] + crop_size[0],
                                top_left_corner[1] + crop_size[1]))

        # Rescale cropped image
        augmented_img = cropped_img.resize(img.size)
        augmented_label = cropped_label.resize(img.size)

    # Elastic transform
    elasticTransfrom = Compose(ElasticTransform(alpha=35, sigma=5, alpha_affine=5, approximate=True, p=0.5))
    aug = elasticTransfrom(image=np.array(augmented_img), mask=np.array(augmented_label))
    augmented_img = Image.fromarray(aug["image"])
    augmented_label = Image.fromarray(aug["mask"])

    # Flip image
    flip_choices = ['horizontal', 'vertical', 'diagonal', 'none']
    flip_choice = random.choice(flip_choices)
    match flip_choice:
        case 'horizontal':
            augmented_img = augmented_img.transpose(method=Image.FLIP_LEFT_RIGHT)
            augmented_label = augmented_label.transpose(method=Image.FLIP_LEFT_RIGHT)
        case 'vertical':
            augmented_img = augmented_img.transpose(method=Image.FLIP_TOP_BOTTOM)
            augmented_label = augmented_label.transpose(method=Image.FLIP_TOP_BOTTOM)
        case 'diagonal':
            augmented_img = augmented_img.transpose(method=Image.TRANSPOSE)
            augmented_label = augmented_label.transpose(method=Image.TRANSPOSE)
        case _:
            pass

    # Convert label to binary
    augmented_label_arr = np.array(augmented_label)
    augmented_label_arr[augmented_label_arr > 128] = 255
    augmented_label_arr[augmented_label_arr <= 128] = 0
    augmented_label = Image.fromarray(augmented_label_arr)

    # Crop image
    augmented_img = crop_around_center(augmented_img, 508, 508)
    augmented_label = crop_around_center(augmented_label, 508, 508)
    return augmented_img, augmented_label


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train_images = glob(f"{input_filepath}/train_images/train*")
    train_labels = glob(f"{input_filepath}/train_labels/labels*")
    iterator = tqdm(enumerate(zip(train_images, train_labels)), total=len(train_labels))
    for i, (train_img, train_label) in iterator:
        # Load images
        img = Image.open(train_img)
        label = Image.open(train_label)

        # Get names
        img_name = train_img.split('\\')[-1].split('.')[0]
        label_name = train_label.split('\\')[-1].split('.')[0]

        # Augment
        for j in range(100):
            augmented_image, augmented_label = augment_image(img, label)
            augmented_image.save(f"{output_filepath}/train_images/{img_name}_{j + 1}.png")
            augmented_label.save(f"{output_filepath}/train_labels/{label_name}_{j + 1}.png")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
