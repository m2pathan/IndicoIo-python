from indicoio.utils.api import api_handler
from indicoio.utils.decorators import detect_batch_decorator


@detect_batch_decorator
def add_data(data, cloud=None, batch=False, api_key=None, version=None, **kwargs):
    """
    This is the basic training endpoint. Given a piece of text and a score, either categorical
    or numeric, this endpoint will train a new model given the additional piece of information.

    Inputs
    data - List: The text and collection/score associated with it. The length of the text (string) should ideally
      be longer than 100 characters and contain at least 10 words. While the API will support
      shorter text, you will find that the accuracy of results improves significantly with longer
      examples. For an additional fee, this end point will support image input as well. The collection/score
      can be a string or float. This is the variable associated with the text. This can either be categorical
      (the tag associated with the post) or numeric (the number of Facebook shares the post
      received). However it can only be one or another within a given label.
    id (optional) - String: This is a unique ID associated with the given piece of text. This can be
      used to later remove a piece of text from the model in case a mistake was made in
      training. The text can also be removed without an ID by simple referencing a copy of the
      text itself, but this will only work if the text matches the original input exactly. If the text
      you work with is liable to change then associating an ID with each example is advised.
    label (optional) - String: This is an identifier for the particular model being trained. The indico
      API allows you to train a number of different models. If the label is not provided, indico
      will add a default label.
    api_key (optional) - String: Your API key, required only if the key has not been declared
      elsewhere. This allows the API to recognize a request as yours and automatically route it
      to the appropriate destination.
    cloud (optional) - String: Your private cloud domain, required only if the key has not been declared
      elsewhere. This allows the API to recognize a request as yours and automatically route it
      to the appropriate destination.

    Example usage:

    .. code-block:: python

       >>> text = "London Underground's boss Mike Brown warned that the strike ..."
       >>> indicoio.train([text, .5])
    """
    batch = isinstance(data[0], list)
    url_params = {"batch": batch, "api_key": api_key, "version": version, method: "add_data"}
    return api_handler(data, cloud=cloud, api="training", url_params=url_params, private=True, **kwargs)


def collections(cloud=None, api_key=None, version=None, **kwargs):
    """
    This is a status report endpoint. It is used to get the status on all of the collections currently trained, as
    well as some basic statistics on their accuracies.

    Inputs
    api_key (optional) - String: Your API key, required only if the key has not been declared
      elsewhere. This allows the API to recognize a request as yours and automatically route it
      to the appropriate destination.
    cloud (optional) - String: Your private cloud domain, required only if the key has not been declared
      elsewhere. This allows the API to recognize a request as yours and automatically route it
      to the appropriate destination.

    Example usage:

    .. code-block:: python

       >>> collections = indicoio.collections()
      {
        "tag_predictor": {
          "type": "categorical",
          "number_of_samples": 224
        }, "popularity_predictor": {
          "type": "regression",
          "number_of_samples": 231
        }
      }
    """
    url_params = {"batch": False, "api_key": api_key, "version": version, "method": "collections"}
    return api_handler(None, cloud=cloud, api="training", url_params=url_params, private=True, **kwargs)


@detect_batch_decorator
def predict(text, cloud=None, batch=False, api_key=None, version=None, **kwargs):
    """
    This is the prediction endpoint. This will be the primary interaction point for all predictive
    analysis.

    Inputs
    text - String: The text example being provided to the API. As a general rule, the text should be as
      similar to the examples given to the train function (above) as possible. Because language
      in different domains is used very differently the accuracy will generally drop as the
      difference between this text and the training text increases.
    sentences (optional) - Boolean: By default the predict function will return analysis on the full document
      that has been passed in. That said, we also have the option to provide a more fine-grained
      analysis by setting this parameter to True. When set to True, the API will return scores for
      each sentence in the provided text.
    label (optional) - String: This is an identifier for the particular model to use for prediction. The
      response format for the given label will match the format of the training examples
    api_key (optional) - String: Your API key, required only if the key has not been declared
      elsewhere. This allows the API to recognize a request as yours and automatically route it
      to the appropriate destination.
    cloud (optional) - String: Your private cloud domain, required only if the key has not been declared
      elsewhere. This allows the API to recognize a request as yours and automatically route it
      to the appropriate destination.

    Example usage:

    .. code-block:: python

      >>> text = "I am Sam. Sam I am."
      >>> prediction = indicoio.predict(text)
      .75
      >>> predictions = indicoio.predict(text, sentences=True)
      {
        "I am Sam.": .62,
        "Sam I am.": .9
      }
    """
    url_params = {"batch": batch, "api_key": api_key, "version": version}
    return api_handler(text, cloud=cloud, api="training", url_params=url_params, private=True, **kwargs)


def clear_collection(collection, cloud=None, api_key=None, version=None, **kwargs):
    """
    This is an API made to remove all of the data associated from a given colletion. If there's been a data
    corruption issue, or a large amount of incorrect data has been fed into the API it is often difficult
    to correct. This allows you to clear a colletion and start from scratch. Use with caution! This is not
    reversible.

    Inputs
    colletion - String: the colletion from which you wish to remove the specified text.
    api_key (optional) - String: Your API key, required only if the key has not been declared
      elsewhere. This allows the API to recognize a request as yours and automatically route it
      to the appropriate destination.
    cloud (optional) - String: Your private cloud domain, required only if the key has not been declared
      elsewhere. This allows the API to recognize a request as yours and automatically route it
      to the appropriate destination.

    Example usage:

    .. code-block:: python

      >>> indicoio.clear_collection("popularity_predictor")

    """
    kwargs['collection'] = collection
    url_params = {"batch": False, "api_key": api_key, "version": version, "method": "clear_collection"}
    return api_handler(None, cloud=cloud, api="training", url_params=url_params, private=True, **kwargs)


@detect_batch_decorator
def remove_example(collection, cloud=None, batch=False, api_key=None, version=None, **kwargs):
    """
    This is an API made to remove a single instance of training data. This is useful in cases where a
    single instance of content has been modified, but the remaining examples remain valid. For
    example, if a piece of content has been retagged.

    Inputs
    id (optional) - String: The unique identifier of the piece of text you wish to remove. This is the
      same id provided above in the train API. This must be provided if text is not, and viceversa.
      text (optional)
    text (optional) - String: The exact text you wish to remove from the given collection. If the string
      provided does not match a known piece of text then this will fail. Again, this is required if
      an id is not provided, and vice-versa.
    collection - String: the collection from which you wish to remove the specified text.
    api_key (optional) - String: Your API key, required only if the key has not been declared
      elsewhere. This allows the API to recognize a request as yours and automatically route it
      to the appropriate destination.
    cloud (optional) - String: Your private cloud domain, required only if the key has not been declared
      elsewhere. This allows the API to recognize a request as yours and automatically route it
      to the appropriate destination.

    Example usage:

    .. code-block:: python

      >>> indicoio.remove_example(id="user202", collection="popularity_predictor")
      >>> indicoio.remove_example(text="I am Sam. Sam I am.", lablel="popularity_predictor")

    """
    kwargs['collection'] = collection
    batch = isinstance((kwargs['text'] if 'text' in kwargs else kwargs['id']), list)
    url_params = {"batch": batch, "api_key": api_key, "version": version, 'method': 'remove_example'}
    return api_handler(None, cloud=cloud, api="training", url_params=url_params, private=True, **kwargs)
