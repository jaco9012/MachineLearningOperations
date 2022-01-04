# -*- coding: utf-8 -*-
import click
import logging
import os
import glob
import torch
import numpy as np
from pathlib import Path
from torch.nn.functional import normalize

from click.decorators import argument
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    train_images = torch.empty((0,784))
    train_labels = torch.empty((0,), dtype=torch.int64)

    for np_name in glob.glob(os.path.join(input_filepath,'*.npz')):
        tmp_data = np.load(np_name)
        new_images = torch.FloatTensor(tmp_data["images"])
        new_images = torch.flatten(new_images, start_dim=1)
        new_labels = torch.LongTensor(tmp_data["labels"])

        if np_name == os.path.join(input_filepath,'test.npz'):
            new_images = normalize(new_images, p=1.0, dim=1)
            torch.save(new_images,os.path.join(output_filepath, 'test_images.pt'))
            torch.save(new_labels,os.path.join(output_filepath, 'test_labels.pt'))
            pass
        else:
            train_images = torch.cat((train_images, new_images))
            train_labels = torch.cat((train_labels, new_labels))
    else:
        train_images = normalize(new_images, p=1.0, dim=1)
        torch.save(train_images,os.path.join(output_filepath, 'train_images.pt'))
        torch.save(train_labels,os.path.join(output_filepath, 'train_labels.pt'))       

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
