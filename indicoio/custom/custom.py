import time

from indicoio.utils.api import api_handler
from indicoio.utils.decorators import detect_batch
from indicoio.utils.image import image_preprocess


class Collection(object):

    def __init__(self, collection, *args, **kwargs):
        self.collection = collection

    def add_data(self, data, cloud=None, batch=False, api_key=None, version=None, **kwargs):
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
        collection (optional) - String: This is an identifier for the particular model being trained. The indico
          API allows you to train a number of different models. If the collection is not provided, indico
          will add a default label.
        domain (optional) - String: This is an identifier that helps determine the appropriate techniques for indico
          to use behind the scenes to train your model.  One of {"standard", "topics"}.
        api_key (optional) - String: Your API key, required only if the key has not been declared
          elsewhere. This allows the API to recognize a request as yours and automatically route it
          to the appropriate destination.
        cloud (optional) - String: Your private cloud domain, required only if the key has not been declared
          elsewhere. This allows the API to recognize a request as yours and automatically route it
          to the appropriate destination.

        Example usage:

        .. code-block:: python

           >>> text = "London Underground's boss Mike Brown warned that the strike ..."
           >>> indicoio.add_data([[text, .5]])
        """
        batch = isinstance(data[0], list)
        if batch:
            X, Y = zip(*data)
            X = image_preprocess(X, batch=batch)
            data = map(list, zip(X, Y))
        else:
            data[0] = image_preprocess(data[0], batch=batch)

        kwargs['collection'] = self.collection
        url_params = {"batch": batch, "api_key": api_key, "version": version, 'method': "add_data"}
        return api_handler(data, cloud=cloud, api="custom", url_params=url_params, **kwargs)


    def train(self, cloud=None, batch=False, api_key=None, version=None, **kwargs):
        """
        This is the basic training endpoint. Given an existing dataset this endpoint will train a model.

        Inputs
        collection - String: the name of the collection to train a model using
        api_key (optional) - String: Your API key, required only if the key has not been declared
          elsewhere. This allows the API to recognize a request as yours and automatically route it
          to the appropriate destination.
        cloud (optional) - String: Your private cloud domain, required only if the key has not been declared
          elsewhere. This allows the API to recognize a request as yours and automatically route it
          to the appropriate destination.

        Example usage:

        .. code-block:: python

           >>> indicoio.train(collection)
        """
        kwargs['collection'] = self.collection
        url_params = {"batch": batch, "api_key": api_key, "version": version, 'method': "train"}
        return api_handler(self.collection, cloud=cloud, api="custom", url_params=url_params, private=True, **kwargs)


    def predict(self, data, cloud=None, batch=False, api_key=None, version=None, **kwargs):
        """
        This is the prediction endpoint. This will be the primary interaction point for all predictive
        analysis.

        Inputs
        data - String: The text example being provided to the API. As a general rule, the data should be as
          similar to the examples given to the train function (above) as possible. Because language
          in different domains is used very differently the accuracy will generally drop as the
          difference between this text and the training text increases. Base64 encoded image data, image urls, and
          text content are all valid.
        domain (optional) - String: This is an identifier that helps determine the appropriate techniques for indico
          to use behind the scenes to train your model.  One of {"standard", "topics"}.
        collection (optional) - String: This is an identifier for the particular model to use for prediction. The
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
        """
        batch = detect_batch(data)
        kwargs['collection'] = self.collection
        data = image_preprocess(data, batch=batch)
        url_params = {"batch": batch, "api_key": api_key, "version": version}
        return api_handler(data, cloud=cloud, api="custom", url_params=url_params, private=True, **kwargs)


    def clear(self, cloud=None, api_key=None, version=None, **kwargs):
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
        kwargs['collection'] = self.collection
        url_params = {"batch": False, "api_key": api_key, "version": version, "method": "clear_collection"}
        return api_handler(None, cloud=cloud, api="custom", url_params=url_params, private=True, **kwargs)

    def remove_example(self, data, cloud=None, batch=False, api_key=None, version=None, **kwargs):
        """
        This is an API made to remove a single instance of training data. This is useful in cases where a
        single instance of content has been modified, but the remaining examples remain valid. For
        example, if a piece of content has been retagged.

        Inputs
        data - String: The exact text you wish to remove from the given collection. If the string
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

          >>> indicoio.remove_example(text="I am Sam. Sam I am.", lablel="popularity_predictor")

        """
        kwargs['collection'] = self.collection
        batch = detect_batch(data)
        data = image_preprocess(data, batch=batch)
        url_params = {"batch": batch, "api_key": api_key, "version": version, 'method': 'remove_example'}
        return api_handler(data, cloud=cloud, api="custom", url_params=url_params, private=True, **kwargs)

    def wait(self, interval=1):
        """
        Block until the collection's model is completed training
        """
        while self.info().get('status') != "ready":
            time.sleep(interval)

    def info(self):
        """
        Return the current state of the model associated with a given collection
        """
        return collections().get(self.collection)


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
            "input_type": "text",
            "model_type": "classification",
            "number_of_samples": 224
            'status': 'ready'
          }, "popularity_predictor": {
            "input_type": "text",
            "model_type": "regression",
            "number_of_samples": 231
            'status': 'training'
          }
        }
      }
    """
    url_params = {"batch": False, "api_key": api_key, "version": version, "method": "collections"}
    return api_handler(None, cloud=cloud, api="custom", url_params=url_params, private=True, **kwargs)
