import json
import mlflow

print('Hello mmdetection')

open('output.json') as output:
    json.dump({
        'Status': 'No output yet'
    }, output)

mlflow.log_artifact('output.json')