# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    X, y = np.load(input_filepath + "training.npy"), np.load(input_filepath + "training_labels.npy")

    # Split your data and labels into a training set and a temporary set that will later be split into a validation
    # and testing set
    train_data_temp, test_data, train_labels_temp, test_labels = train_test_split(X, y, test_size=0.1,
                                                                                  random_state=42, shuffle=True)
    # Split the temporary set into a validation set and a final testing set
    train_data, val_data, train_labels, val_labels = train_test_split(train_data_temp, train_labels_temp,
                                                                      test_size=0.111, random_state=42, shuffle=True)

    # Print the sizes of each set
    print(f"Training set size: {len(train_data)}")
    print(f"Training labels size: {len(train_labels)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Validation labels size: {len(val_labels)}")
    print(f"Testing set size: {len(test_data)}")
    print(f"Testing labels size: {len(test_labels)}")

    np.save(output_filepath + "training.npy", train_data)
    np.save(output_filepath + "training_labels.npy", train_labels)

    np.save(output_filepath + "test.npy", test_data)
    np.save(output_filepath + "test_labels.npy", test_labels)

    np.save(output_filepath + "val.npy", val_data)
    np.save(output_filepath + "val_labels.npy", val_labels)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
