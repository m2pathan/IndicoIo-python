from functools import wraps, partial
import warnings

Version, version, __version__, VERSION = ('0.10.3',) * 4

JSON_HEADERS = {
    'Content-type': 'application/json',
    'Accept': 'application/json',
    'client-lib': 'python',
    'version-number': VERSION
}

from indicoio.text.twitter_engagement import twitter_engagement
from indicoio.text.sentiment import political, posneg, sentiment_hq
from indicoio.text.sentiment import posneg as sentiment
from indicoio.text.lang import language
from indicoio.text.tagging import text_tags
from indicoio.text.keywords import keywords
from indicoio.text.ner import named_entities
from indicoio.text.self_train_classifier import add_data, collections, predict, clear_collection, remove_example
from indicoio.images.fer import fer
from indicoio.images.features import facial_features, image_features
from indicoio.images.faciallocalization import facial_localization
from indicoio.images.recognition import image_recognition
from indicoio.images.filtering import content_filtering
from indicoio.utils.multi import analyze_image, analyze_text, intersections

from indicoio.config import API_NAMES

def deprecation_decorator(f, api):
    @wraps(f)
    def wrapper(*args, **kwargs):
        warnings.warn(
            "'batch_" + api + "' will be deprecated in the next major update.  Please call '" + api + "' instead with the same arguments.",
            DeprecationWarning
        )
        return f(*args, **kwargs)
    return wrapper

def detect_batch_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], list):
            kwargs['batch'] = True
        return f(*args, **kwargs)
    return wrapper

apis = dict((api, globals().get(api)) for api in API_NAMES)

for api in apis:
    globals()[api] = detect_batch_decorator(apis[api])
    globals()['batch_' + api] = partial(deprecation_decorator(apis[api], api), batch=True)
