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

result = model.predict(source)

print('!!!Performed prediction!!!')

with open('output.json', 'w') as output:
    json.dump({
        'Status': 'No output yet'
    }, output)

mlflow.log_artifact('output.json')
