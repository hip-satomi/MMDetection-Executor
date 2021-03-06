import unittest
import os


class TestOnlineSegmentation(unittest.TestCase):

    def setUp(self):
        # download the image
        import requests

        url = 'https://fz-juelich.sciebo.de/s/wAXbC0MoN1G3ST7/download'
        r = requests.get(url, allow_redirects=True)

        print(len(r.content))

        with open('test.png', 'wb') as file:
            file.write(r.content)

    def test_standard(self):
        # test entrypoints: main (Cellpose)
        self.predict('main')

    def predict(self, entrypoint):
        import requests
        from io import BytesIO
        from PIL import Image
        import json

        contours = []

        image = Image.open('test.png')

        # convert image into a binary png stream
        byte_io = BytesIO()
        image.save(byte_io, "png")
        byte_io.seek(0)

        # pack this into form data
        multipart_form_data = [
            ("files", ("data.png", byte_io, "image/png"))
        ]

        # get job specific environment variables
        CI_COMMIT_SHA = os.environ['CI_COMMIT_SHA']
        CI_REPOSITORY_URL = os.environ['CI_REPOSITORY_URL']

        additional_parameters = {
            "config": "https://fz-juelich.sciebo.de/s/LdVbQhCUMNUBnXy/download",
            "checkpoint": "https://fz-juelich.sciebo.de/s/Qngmf7FR7v7GZfS/download"
        }

        # exactly request segmentation with the current repo version
        params = dict(
            repo=CI_REPOSITORY_URL,
            entry_point=entrypoint,
            version=CI_COMMIT_SHA,
            parameters=json.dumps(additional_parameters),
        )

        # send a request to the server
        response = requests.post(
            'http://segserve/batch-image-prediction/', params=params, files=multipart_form_data, timeout=20 * 60
        )

        # output response
        print(response.content)
        print(response.status_code)

        # the request should be successful
        self.assertTrue(response.status_code == 200)

if __name__ == '__main__':
    unittest.main()