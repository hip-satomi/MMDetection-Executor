import json
import mlflow

print('Hello mmdetection')

with open('output.json', 'w') as output:
    json.dump({
        'Status': 'No output yet'
    }, output)

mlflow.log_artifact('output.json')
