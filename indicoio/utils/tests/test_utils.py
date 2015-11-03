import sys
import unittest

from mock import patch, MagicMock

from indicoio.utils import is_url
from indicoio.utils.errors import IndicoError
from indicoio import config

mock_response = MagicMock()
mock_response.headers = {
    'x-warning': 'testing warning'
}
mock_response.status_code = 200
mock_response.json = MagicMock(return_value={'results': 0.5})

class UtilsTest(unittest.TestCase):
    def test_is_urls(self):
        boring_image = [0]*(32**2)
        boring_images = [boring_image]*100

        assert not is_url(boring_image, batch=False)
        assert not is_url(boring_images, batch=True)

        url = 'http://picturepicture.com/picture'
        urls = [url]*100

        assert is_url(url, batch=False)
        assert is_url(urls, batch=True)

    @patch('indicoio.utils.api.warnings.warn')
    @patch('indicoio.utils.api.requests.post', MagicMock(return_value=mock_response))
    def test_api_handler(self, mock_warn):
        from indicoio.utils.api import api_handler
        api_handler("test", cloud=None, api='sentiment')
        assert mock_warn.called_with(mock_response.headers.get('x-warning'))

        if config.cloud:
            api_handler("test", cloud=None, api='sentiment', private=True)
        else:
            with self.assertRaises(IndicoError) as context:
                api_handler("test", cloud=None, api='sentiment', private=True)

            self.assertTrue("Api 'sentiment' is only available on private cloud" in context.exception)
