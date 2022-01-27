import unittest
from main import main, parse_args
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

if __name__ == '__main__':
    unittest.main()