import json
import mlflow
from acia.segm.processor.offline import OfflineModel
import sys

config = sys.argv[2]
checkpoint = sys.argv[4]

print('Loading model...')
model = OfflineModel(config, checkpoint)
print('Done')

print('Hello mmdetection')

with open('output.json', 'w') as output:
    json.dump({
        'Status': 'No output yet'
    }, output)

mlflow.log_artifact('output.json')
