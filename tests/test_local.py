import unittest
from main import main, parse_args
import mlflow
import json

from git_utils import get_git_revision_short_hash, get_git_url

class TestLocal(unittest.TestCase):

    def setUp(self):
        # download the image
        import requests

        url = 'https://fz-juelich.sciebo.de/s/wAXbC0MoN1G3ST7/download'
        r = requests.get(url, allow_redirects=True)

        print(len(r.content))

        with open('test.jpeg', 'wb') as file:
            file.write(r.content)

    def test_prediction(self):
        main(parse_args(["--config", "https://fz-juelich.sciebo.de/s/LdVbQhCUMNUBnXy/download", "--checkpoint", "https://fz-juelich.sciebo.de/s/Qngmf7FR7v7GZfS/download", "test.jpeg"]))

    def test_mlproject_info(self):
        run = mlflow.projects.run(
            './',
            entry_point="info",
            backend='local',
        )

        # download the output artifact
        client = mlflow.tracking.MlflowClient()
        client.download_artifacts(run.run_id, 'output.json', './')

        with open('output.json', 'r') as input_file:
            info_result = json.load(input_file)
            self.assertTrue(info_result['name'] == 'mmdetection-executor')
            self.assertTrue(info_result['git_hash'] == get_git_revision_short_hash())
            self.assertTrue(info_result["git_url"] == get_git_url())
            self.assertTrue(info_result["type"] == "info")


if __name__ == '__main__':
    unittest.main()