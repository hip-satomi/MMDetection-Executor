# MMDetection-Executor

This is an mlproject wrapper around the [mmdetection](https://github.com/open-mmlab/mmdetection) segmentation and detection framework supporting a wide variety of machine learing approaches.

## Local testing

Make sure you have [anaconda]() installed and an active environment with `mlflow`. Then execute
```bash
pip install mlflow
mlproject run ./ -e main -P input_images=<path to your local image or image folder (*.png)> -P config=<path/url to your mmdetection config> -P checkpoint=<path/url to your mmdetection model checkpoint>
```
The resulting segmentation should be written to `output.json` and logged as an artifact in the mlflow run.

### Caching
By default the config and checkpoint paths are cached when specified as a url. Therefore, the `CACHE_FOLDER` environment variable must point to an existing folder that can be used for caching the files.

## Intended Usage

The wrapper is used to deploy any mmdetection methods in the segServe runtime environment. SegServe can be used to host 3rd party segmentation algorithms and execute them on a central computer while providing a REST interface for clients.
