from indicoio.utils.api import api_handler
from indicoio.utils.decorators import detect_batch_decorator


@detect_batch_decorator
def keywords(text, cloud=None, batch=False, api_key=None, version=None, **kwargs):
    """
    Given input text, returns series of keywords and associated scores

    Example usage:

    .. code-block:: python

       >>> import indicoio
       >>> import numpy as np
       >>> text = 'Monday: Delightful with mostly sunny skies. Highs in the low 70s.'
       >>> keywords = indicoio.keywords(text, top_n=3)
       >>> print "The keywords are: "+str(keywords.keys())
       u'The keywords are ['delightful', 'highs', 'skies']

    :param text: The text to be analyzed.
    :type text: str or unicode
    :rtype: Dictionary of feature score pairs
    """
    url_params = {'batch': batch, 'api_key': api_key}
    return api_handler(text, cloud=cloud, api="keywords", url_params=url_params, **kwargs)
