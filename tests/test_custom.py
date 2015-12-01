import unittest
from indicoio.utils.errors import IndicoError
from indicoio.custom import Collection, collections
import time

collection_name = "__test_python_text__"
test_data = [['input 1', 'label 1'], ['input 2', 'label 2'], ['input 3', 'label 3'], ['input 4', 'label 4']]

image_collection_name = "__test_python_image__"
image_test_data = [
    ["https://i.imgur.com/xUX1rvY.png", "dog"],
    ["https://i.imgur.com/xUX1rvY.png", "dog"],
    ['https://i.imgur.com/2Q0EWRz.jpg', 'cat'],
    ['https://i.imgur.com/XhUDCMP.jpg', 'cat']
]


class CustomAPIsTestCase(unittest.TestCase):

    def tearDown(self):
        try:
            Collection(collection_name).clear()
        except IndicoError:
            pass

        try:
            Collection(image_collection_name).clear()
        except IndicoError:
            pass

    def test_add_predict(self):
        collection = Collection(collection_name)
        collection.add_data(test_data)
        collection.train()
        collection.wait()
        result = collection.predict(test_data[0][0])
        assert test_data[0][1] in result.keys()

    def test_list_collection(self):
        collection = Collection(collection_name)
        collection.add_data(test_data)
        collection.train()
        collection.wait()
        assert collections()[collection_name]

    def test_add_large_batch(self):
        collection = Collection(collection_name)
        collection.add_data(test_data*100)
        collection.train()
        collection.wait()
        result = collection.predict(test_data[0][0])
        assert test_data[0][1] in result.keys()

    def test_add_image_batch(self):
        collection = Collection(image_collection_name)
        collection.add_data(image_test_data)
        collection.train()
        collection.wait()
        result = collection.predict(image_test_data[0][0])
        assert image_test_data[0][1] in result.keys()

    def test_clear_example(self):
        collection = Collection(collection_name)
        collection.add_data(test_data)
        collection.train()
        collection.wait()
        result = collection.predict(test_data[0][0])
        assert test_data[0][1] in result.keys()
        collection.remove_example(test_data[0][0])
        collection.train()
        collection.wait()
        result = collection.predict(test_data[0][0])
        assert test_data[0][1] not in result.keys()

    def test_clear_collection(self):
        collection = Collection(collection_name)
        collection.add_data(test_data)
        collection.train()
        collection.wait()
        assert collections()[collection_name]
        collection.clear()
        assert not collections().get(collection_name)
