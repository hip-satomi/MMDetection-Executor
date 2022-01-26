import json
import mlflow
from acia.segm.processor.offline import OfflineModel
from acia.segm.local import LocalImageSource
import sys
from urllib.parse import urlparse
import torch

from utils import get_git_revision_short_hash, get_git_url


import argparse
import os
import glob

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('images', type=str, nargs='+',
                    help='list of images')
parser.add_argument('--config', type=str, help="mmdetection model configuration file.")
parser.add_argument('--checkpoint', type=str, help="mmdetection model checkpoint file.")
parser.add_argument('--package', type=str, help="Zip Packaged checkpoint and config file (checkpoint.pth, config.py)")
parser.add_argument('--cached', type=bool, default=True, action="store_true", help="Whether to try to use a cached version file")

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

# TODO: take a package

images = [LocalImageSource(image_path) for image_path in args.images]

input_image = sys.argv[2]
config = sys.argv[4]
checkpoint = sys.argv[6]

print(f"Running prediction on {input_image}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Loading model to {device}...')
model = OfflineModel(config, checkpoint, device=device)
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
