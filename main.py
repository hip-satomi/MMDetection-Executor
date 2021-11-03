import json
import mlflow
from acia.segm.processor.offline import OfflineModel
from acia.segm.local import LocalImageSource
import sys
import subprocess
from urllib.parse import urlparse


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def get_git_url() -> str:
    basic_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode('ascii').strip()
    parsed = urlparse(basic_url)
    if parsed.username and parsed.password:
        # erase username and password
        return parsed._replace(netloc="{}".format(parsed.hostname)).geturl()
    else:
        return parsed.geturl()

input_image = sys.argv[2]
config = sys.argv[4]
checkpoint = sys.argv[6]

print(f"Running prediction on {input_image}")

source = LocalImageSource(input_image)

print('Loading model...')
model = OfflineModel(config, checkpoint)
print('Done')

overlay = model.predict(source)

print(f'Detected {len(overlay)} cells!')

detections = [dict(
    score = det.score,
    contour_coordinates = det.coordinates.tolist(),
    label = "Cell",
    type = 'Polygon'

 ) for det in overlay]
# get the git hash of the current commit
short_hash = get_git_revision_short_hash()
git_url = get_git_url()

result = dict(
    model_version = f'{git_url}#{short_hash}',
    format_version = '0.1',
    segmentation = detections
}

print('!!!Performed prediction!!!')

with open('output.json', 'w') as output:
    json.dump(output_dict, result)

mlflow.log_artifact('output.json')
