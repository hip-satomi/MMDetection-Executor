import json
import mlflow
from acia.segm.processor.offline import OfflineModel
from acia.segm.local import LocalImageSource
import sys

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

detections = [{
    "score": det.score,
    "coordinates": det.coordinates,
    "label": "Cell"
} for det in overlay]

output_dict = {
    'model_version': '0.x.x',
    'segmentation': detections
}

print('!!!Performed prediction!!!')

with open('output.json', 'w') as output:
    json.dump(output_dict, output)

mlflow.log_artifact('output.json')
