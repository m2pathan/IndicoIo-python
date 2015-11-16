from indicoio.config import TEXT_APIS, IMAGE_APIS, API_NAMES
from indicoio.utils.api import api_handler
from indicoio.utils.image import image_preprocess
from indicoio.utils.errors import IndicoError
from indicoio.utils.decorators import detect_batch_decorator


CLIENT_SERVER_MAP = dict((api, api.strip().replace("_", "").lower()) for api in API_NAMES)
SERVER_CLIENT_MAP = dict((v, k) for k, v in CLIENT_SERVER_MAP.items())
AVAILABLE_APIS = {
    'text': TEXT_APIS,
    'image': IMAGE_APIS
}

def invert_dictionary(d):
    return {
        element: key for key, values in d.items()
        for element in values
    }

API_TYPES = invert_dictionary(AVAILABLE_APIS)


@detect_batch_decorator
def intersections(data, apis = None, **kwargs):
    """
    Helper to make multi requests of different types.

    :param data: Data to be sent in API request
    :param type: String type of API request
    :rtype: Dictionary of api responses
    """
    # Client side api name checking

    # remove auto-inserted batch param
    kwargs.pop('batch', None)

    if not isinstance(apis, list) or len(apis) != 2:
        raise IndicoError("Argument 'apis' must be of length 2")
    if isinstance(data, list) and len(data) < 3:
        raise IndicoError(
            "At least 3 examples are required to use the intersections API"
        )

    api_types = list(map(API_TYPES.get, apis))
    if api_types[0] != api_types[1]:
        raise IndicoError(
            "Both `apis` must accept the same kind of input to use the intersections API"
        )

    cloud = kwargs.pop("cloud", None)

    url_params = {
        'batch': False,
        'api_key': kwargs.pop('api_key', None),
        'apis': apis
    }

    return api_handler(data, cloud=cloud, api="apis/intersections", url_params=url_params, **kwargs)


def multi(data, datatype, apis, batch=False, **kwargs):
    """
    Helper to make multi requests of different types.

    :param data: Data to be sent in API request
    :param datatype: String type of API request
    :param apis: List of apis to use.
    :param batch: Is this a batch request?
    :rtype: Dictionary of api responses
    """
    # Client side api name checking - strictly only accept func name api
    available = AVAILABLE_APIS.get(datatype)
    invalid_apis = [api for api in apis if api not in available]
    if invalid_apis:
        raise IndicoError(
            "%s are not valid %s APIs. Please reference the available APIs below:\n%s"
            % (", ".join(invalid_apis), datatype, ", ".join(available))
        )

    # Convert client api names to server names before sending request
    cloud = kwargs.pop("cloud", None)
    api_key = kwargs.pop('api_key', None)
    result = api_handler(
        data,
        cloud=cloud,
        api='apis/multiapi',
        url_params={
            "apis":apis,
            "batch":batch,
            "api_key":api_key
        },
        **kwargs
    )
    return handle_response(result)


def handle_response(result):
    # Parse out the results to a dicionary of api: result
    return dict((api, parsed_response(api, res))
        for api, res in result.items())


@detect_batch_decorator
def analyze_text(input_text, apis=TEXT_APIS, **kwargs):
    """
    Given input text, returns the results of specified text apis. Possible apis
    include: [ 'text_tags', 'political', 'sentiment', 'language' ]

    Example usage:

    .. code-block:: python

       >>> import indicoio
       >>> text = 'Monday: Delightful with mostly sunny skies. Highs in the low 70s.'
       >>> results = indicoio.analyze_text(data = text, apis = ["language", "sentiment"])
       >>> language_results = results["language"]
       >>> sentiment_results = results["sentiment"]

    :param text: The text to be analyzed.
    :param apis: List of apis to use.
    :type text: str or unicode
    :type apis: list of str
    :rtype: Dictionary of api responses
    """

    cloud = kwargs.pop('cloud', None)
    batch = kwargs.pop('batch', False)
    api_key = kwargs.pop('api_key', None)

    return multi(
        data=input_text,
        datatype="text",
        cloud=cloud,
        batch=batch,
        api_key=api_key,
        apis=apis,
        **kwargs
    )


@detect_batch_decorator
def analyze_image(image, apis=IMAGE_APIS, **kwargs):
    """
    Given input image, returns the results of specified image apis. Possible apis
    include: ['fer', 'facial_features', 'image_features']

    Example usage:

    .. code-block:: python

       >>> import indicoio
       >>> import numpy as np
       >>> face = np.zeros((48,48)).tolist()
       >>> results = indicoio.analyze_image(image = face, apis = ["fer", "facial_features"])
       >>> fer = results["fer"]
       >>> facial_features = results["facial_features"]

    :param text: The text to be analyzed.
    :param apis: List of apis to use.
    :type text: str or unicode
    :type apis: list of str
    :rtype: Dictionary of api responses
    """

    cloud = kwargs.pop('cloud', None)
    batch = kwargs.pop('batch', False)
    api_key = kwargs.pop('api_key', None)

    return multi(
        data=image_preprocess(image, batch=batch),
        datatype="image",
        cloud=cloud,
        batch=batch,
        api_key=api_key,
        apis=apis,
        **kwargs
    )

def parsed_response(api, response):
    result = response.get('results', False)
    if result != False:
        return result
    raise IndicoError(
        "Sorry, the %s API returned an unexpected response.\n\t%s"
        % (api, response.get('error', ""))
    )
