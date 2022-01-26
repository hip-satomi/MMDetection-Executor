import json
from multiprocessing.sharedctypes import Value
import mlflow
from acia.segm.processor.offline import OfflineModel
from acia.segm.local import LocalImageSource
import sys
from urllib.parse import urlparse
import torch
import validators
import hashlib
import tempfile

from utils import CACHE_FOLDER, cached_file, get_git_revision_short_hash, get_git_url, is_cached_file


import argparse
import os
import glob

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('images', type=str, nargs='+',
                    help='list of images')
parser.add_argument('--config', type=str, help="mmdetection model configuration file.")
parser.add_argument('--checkpoint', type=str, help="mmdetection model checkpoint file.")
parser.add_argument('--package', type=str, help="Zip Packaged checkpoint and config file (checkpoint.pth, config.py)")
parser.add_argument('--cached', type=bool, default=True, help="Whether to try to use a cached version file")

args = parser.parse_args()

# expand to array of images (e.g. if the path is a folder)
if len(args.images) == 1:
    image_path = args.images[0]
    if os.path.isdir(image_path):
        # it's a folder, iterate all images in the folder
        args.images = sorted(glob.glob(os.path.join(image_path, '*.png')))
    else:
        # it may be a list of images
        args.images = image_path.split(' ')

with tempfile.TemporaryDirectory() as tmpcache:
    #elif args.config and args.checkpoint:
    if not args.cached:
        cache_folder = tmpcache
    else:
        cache_folder = CACHE_FOLDER

    # get the cached versions
    config_path = cached_file(args.config, cache_folder=cache_folder)
    checkpoint_path = cached_file(args.checkpoint, cache_folder=cache_folder)

    # load the images
    images = [LocalImageSource(image_path) for image_path in args.images]

    print(f"Running prediction on {image_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Loading model to {device}...')
    model = OfflineModel(config_path, checkpoint_path, device=device)
    print('Done')

    # get the git hash of the current commit
    short_hash = get_git_revision_short_hash()
    git_url = get_git_url()

    full_result = []

    for source in images:
        overlay = model.predict(source)

        print(f'Detected {len(overlay)} cells!')

        detections = [dict(
            score = det.score,
            contour_coordinates = det.coordinates.tolist(),
            label = "Cell",
            type = 'Polygon'

        ) for det in overlay]

        result = dict(
            model_version = f'{git_url}#{short_hash}',
            format_version = '0.1',
            segmentation = detections
        )

        full_result.append(result)

    print('!!!Performed prediction!!!')

    if len(images) == 1:
        full_result = full_result[0]


    with open('output.json', 'w') as output:
        json.dump(full_result, output)

    mlflow.log_artifact('output.json')
