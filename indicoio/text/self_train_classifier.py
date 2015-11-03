from indicoio.utils.api import api_handler

def train(text, cloud=None, batch=False, api_key=None, version=None, **kwargs):
    """
    This is the basic training endpoint. Given a piece of text and a score, either categorical
    or numeric, this endpoint will train a new model given the additional piece of information.

    Inputs
    text - String: The text example being provided to the API. The length of this string should ideally
      be longer than 100 characters and contain at least 10 words. While the API will support
      shorter text, you will find that the accuracy of results improves significantly with longer
      examples. For an additional fee, this end point will support image input as well.
    score - String | Float: This is the variable associated with the text. This can either be categorical
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
       >>> entities = indicoio.train(text, .5)
    """
    url_params = {"batch": batch, "api_key": api_key, "version": version}
    return api_handler(text, cloud=cloud, api="train", url_params=url_params, private=True, **kwargs)

def labels(cloud=None, api_key=None, version=None, **kwargs):
    """
    This is a status report endpoint. It is used to get the status on all of the labels currently trained, as
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
       >>> entities = indicoio.train()
      {
        "tag_predictor": {
          "type": "categorical",
          "number_of_samples": 224,
          "accuracy": 0.34
        }, "popularity_predictor": {
          "type": "regression",
          "number_of_samples": 231,
          "mean_error": 21.2
        }
      }
    """
    url_params = {"batch": False, "api_key": api_key, "version": version}
    return api_handler(None, cloud=cloud, api="labels", url_params=url_params, private=True, **kwargs)
