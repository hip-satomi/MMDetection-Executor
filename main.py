import json
from multiprocessing.sharedctypes import Value
from typing import Sequence
import mlflow
from acia.segm.processor.offline import OfflineModel
from acia.segm.local import LocalImageSource
import sys
from urllib.parse import urlparse
import torch
import tempfile
import sys

from utils import CACHE_FOLDER, cached_file, get_git_revision_short_hash, get_git_url, is_cached_file


import argparse
import os
import glob

def parse_args(arguments: Sequence[str]):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('images', type=str, nargs='+',
                        help='list of images')
    parser.add_argument('--config', type=str, required=True, help="mmdetection model configuration file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="mmdetection model checkpoint file.")
    parser.add_argument('--package', type=str, help="Zip Packaged checkpoint and config file (checkpoint.pth, config.py)")
    parser.add_argument('--cached', type=bool, default=True, help="Whether to try to use a cached version file")

    return parser.parse_args(arguments)

def main(args):

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
        config_path = cached_file(args.config, cache_folder=cache_folder, enforce_ending='.py')
        checkpoint_path = cached_file(args.checkpoint, cache_folder=cache_folder)

        # load the images
        images = [LocalImageSource(image_path) for image_path in args.images]

        print(f"Running prediction on {''.join(args.images)}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Loading model to {device}...')
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

            full_result.append(detections)

        print('!!!Performed prediction!!!')

        result = dict(
            model = f'{git_url}#{short_hash}',
            format_version = '0.2',
            segmentation_data = full_result
        )

        with open('output.json', 'w') as output:
            json.dump(result, output)

        mlflow.log_artifact('output.json')

if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))

