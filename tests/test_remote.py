#!/usr/bin/python
# -*- coding: utf-8 -*-
import unittest
import os, random
from PIL import Image
from requests import ConnectionError

from nose.plugins.skip import Skip, SkipTest
from six import PY3

from indicoio import config
from indicoio import political, sentiment, fer, facial_features, facial_localization, content_filtering, language, image_features, text_tags
from indicoio import keywords, sentiment_hq, twitter_engagement, named_entities, intersections, analyze_image, analyze_text
from indicoio import personas, personality
from indicoio.utils.errors import IndicoError

DIR = os.path.dirname(os.path.realpath(__file__))

class BatchAPIRun(unittest.TestCase):

    def setUp(self):
        self.api_key = config.api_key
        config.url_protocol = "http:"

        if not all(self.api_key):
            raise SkipTest

    def tearDown(self):
        config.url_protocol = "https:"

    def test_batch_texttags(self):
        test_data = ["On Monday, president Barack Obama will be..."]
        response = text_tags(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))

    def test_batch_keywords(self):
        test_data = ["A working api is key to the success of our young company"]
        words = [set(text.lower().split()) for text in test_data]
        response = keywords(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))
        self.assertTrue(set(response[0].keys()).issubset(words[0]))

    def test_batch_posneg(self):
        test_data = ['Worst song ever', 'Best song ever']
        response = sentiment(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))
        self.assertTrue(response[0] < 0.5)

    def test_batch_sentiment_hq(self):
        test_data = ['Worst song ever', 'Best song ever']
        response = sentiment_hq(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))
        self.assertTrue(response[0] < 0.5)

    def test_batch_twitter_engagement(self):
        test_string = "Worst song ever."
        response = twitter_engagement([test_string, test_string])

        self.assertTrue(isinstance(response, list))
        self.assertIsInstance(response[0], float)
        self.assertEqual(response[0], response[1])

    def test_batch_personality(self):
        test_string = "I love my friends!"
        response = personality([test_string,test_string])
        categories = ['extraversion', 'openness', 'agreeableness', 'conscientiousness']
        self.assertTrue(isinstance(response, list))
        self.assertIsInstance(response[0]["extraversion"], float)
        for category in categories:
            assert category in response[0].keys()
        self.assertEqual(response[0]["extraversion"], response[1]["extraversion"])

    def test_batch_personas(self):
        test_string = "I love my friends!"
        response = personas([test_string,test_string])
        self.assertTrue(isinstance(response, list))
        self.assertIsInstance(response[0]["commander"], float)
        self.assertEqual(response[0]["commander"], response[1]["commander"])

    def test_batch_political(self):
        test_data = ["Guns don't kill people, people kill people."]
        response = political(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))

    def test_url_support(self):
        test_url = "https://s3-us-west-2.amazonaws.com/indico-test-data/face.jpg"
        response = fer(test_url, api_key=self.api_key)
        self.assertTrue(isinstance(response, dict))
        self.assertEqual(len(response.keys()), 6)

    def test_batch_fer(self):
        test_data = [os.path.normpath(os.path.join(DIR, "data/48by48.png"))]
        response = fer(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))
        self.assertTrue(isinstance(response[0], dict))

    def test_batch_content_filtering(self):
        test_data = [os.path.normpath(os.path.join(DIR, "data/48by48.png"))]
        response = content_filtering(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))
        self.assertTrue(isinstance(response[0], float))

    def test_batch_fer_bad_b64(self):
        test_data = ["$bad#FI jeaf9(#0"]
        self.assertRaises(IndicoError, fer, test_data, api_key=self.api_key)

    def test_batch_fer_good_b64(self):
        test_data = ["iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAg5JREFUeNrEV4uNgzAMpegGyAgZgQ3KBscIjMAGx03QEdqbgG5AOwG3AWwAnSCXqLZkuUkwhfYsvaLm5xc7sZ1dIhdtUVjsLZRFTvp+LSaLq8UZ/s+KMSbZCcY5RV9E4QQKHG7QtgeCGv4PFt8WpzkCcztu3TiL0eJgkQmsVFn0MK+LzYkRKEGpG1GDyZdKRdaolhAoJewXnJsO1jtKCFDlChZAFxyJj2PnBRU20KZg7oMlOAENijpi8hwmGkKkZW2GzONtVLA/DxHAhTO2I7MCVBSQ6nGDlEBJDhyVYiUBHXBxzQm0wE4FzPYsGs856dA9SAAP2oENzFYqR6iAFQpHIAUzO/nxnOgthF/lM3w/3U8KYXTwxG/1IgIulF+wPQUXDMl75UoJZIHstRWpaGb8IGYqwBoKlG/lgpzoUEBoj50p8QtVrmHgaaXyC/H3BFC+e9kGFlCB0CtBF7FifQ8D9zjQQHj0pdOM3F1pUBoFKdxtqkMClScHJCSDlSxhHSNRT5K+FaZnHglrz+AGoxZLKNLYH6s3CkkuyJlp58wviZ4PuSCWDXl5hmjZtxcSCGbDUD3gK7EMOZBLCETrgVBF5K0lI5bIZ0wfrYh8NWHIAiNTPHpuTOKpCes1VTFaiNaFdGwPfdmaqlj6LmjJbgoSSfUW74K3voz+/W0oIeB7HWu2s+dfx3N+eLX8CTAAwUmKjK/dHS4AAAAASUVORK5CYII="]
        response = fer(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))
        self.assertTrue(isinstance(response[0], dict))

    def test_batch_fer_filepath(self):
        test_data = [os.path.normpath(os.path.join(DIR, "data/fear.png"))]
        response = fer(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))
        self.assertTrue(isinstance(response[0], dict))

    def test_fer_detect(self):
        test_data = os.path.normpath(os.path.join(DIR, "data/fear.png"))
        response = fer(test_data, api_key=self.api_key, detect=True)
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        self.assertIn("location", response[0])

    def test_batch_fer_pil_image(self):
        test_data = [Image.open(os.path.normpath(os.path.join(DIR, "data/fear.png")))]
        response = fer(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))
        self.assertTrue(isinstance(response[0], dict))

    def test_batch_fer_nonexistant_filepath(self):
        test_data = ["data/unhappy.png"]
        self.assertRaises(IndicoError, fer, test_data, api_key=self.api_key)

    def test_batch_facial_features(self):
        test_data = [os.path.normpath(os.path.join(DIR, "data/48by48.png"))]
        response = facial_features(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))
        self.assertTrue(isinstance(response[0], list))
        self.assertEqual(len(response[0]), 48)

    def test_batch_image_features_greyscale(self):
        test_data = [os.path.normpath(os.path.join(DIR, "data/48by48.png"))]
        response = image_features(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))
        self.assertTrue(isinstance(response[0], list))
        self.assertEqual(len(response[0]), 4096)

    def test_batch_image_features_rgb(self):
        test_data = [os.path.normpath(os.path.join(DIR, "data/48by48rgb.png"))]
        response = image_features(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))
        self.assertTrue(isinstance(response[0], list))
        self.assertEqual(len(response[0]), 4096)

    def test_batch_language(self):
        test_data = ['clearly an english sentence']
        response = language(test_data, api_key=self.api_key)
        self.assertTrue(isinstance(response, list))
        self.assertTrue(response[0]['English'] > 0.25)

    def test_batch_named_entities(self):
        batch = ["London Underground's boss Mike Brown warned that the strike ..."]
        expected_entities = ("London Underground", "Mike Brown")
        expected_keys = set(["categories", "confidence"])
        entities = named_entities(batch)[0]
        for entity in expected_entities:
            assert entity in expected_entities
            assert not (set(entities[entity]) - expected_keys)

    def test_batch_multi_api_image(self):
        test_data = [os.path.normpath(os.path.join(DIR, "data/48by48.png")),
                     os.path.normpath(os.path.join(DIR, "data/48by48.png"))]
        response = analyze_image(test_data, apis=config.IMAGE_APIS, api_key=self.api_key)

        self.assertTrue(isinstance(response, dict))
        self.assertTrue(set(response.keys()) == set(config.IMAGE_APIS))
        self.assertTrue(isinstance(response["fer"], list))

    def test_batch_multi_api_text(self):
        test_data = ['clearly an english sentence']
        response = analyze_text(test_data, apis=config.TEXT_APIS, api_key=self.api_key)

        self.assertTrue(isinstance(response, dict))
        self.assertTrue(set(response.keys()) == set(config.TEXT_APIS))

    def test_default_multi_api_text(self):
        test_data = ['clearly an english sentence']
        response = analyze_text(test_data, api_key=self.api_key)

        self.assertTrue(isinstance(response, dict))
        self.assertTrue(set(response.keys()) == set(config.TEXT_APIS))

    def test_multi_api_bad_api(self):
        self.assertRaises(IndicoError,
                          analyze_text,
                          "this shouldn't work",
                          apis=["sentiment", "somethingbad"])

    def test_multi_bad_mixed_api(self):
        self.assertRaises(IndicoError,
                            analyze_text,
                            "this shouldn't work",
                            apis=["fer", "sentiment", "facial_features"])

    def test_batch_multi_bad_mixed_api(self):
        self.assertRaises(IndicoError,
                            analyze_text,
                            ["this shouldn't work"],
                            apis=["fer", "sentiment", "facial_features"])

    def test_batch_set_cloud(self):
        test_data = ['clearly an english sentence']
        self.assertRaises(ConnectionError,
                          language,
                          test_data,
                          api_key=self.api_key,
                          cloud='invalid/cloud')


class FullAPIRun(unittest.TestCase):

    def setUp(self):
        self.api_key = config.api_key

    def check_range(self, _list, minimum=0.9, maximum=0.1, span=0.5):
        vector = list(flatten(_list))
        _max = max(vector)
        _min = min(vector)
        self.assertTrue(max(vector) > maximum)
        self.assertTrue(min(vector) < minimum)
        self.assertTrue(_max - _min > span)

    def test_text_tags(self):
        text = "On Monday, president Barack Obama will be..."
        results = text_tags(text)
        max_keys = sorted(results.keys(), key=lambda x:results.get(x), reverse=True)
        assert 'political_discussion' in max_keys[:5]
        results = text_tags(text, top_n=5)
        assert len(results) is 5
        results = text_tags(text, threshold=0.1)
        for v in results.values():
            assert v >= 0.1

    def test_keywords(self):
        text = "A working api is key to the success of our young company"
        words = set(text.lower().split())

        results = keywords(text)
        sorted_results = sorted(results.keys(), key=lambda x:results.get(x), reverse=True)
        assert 'api' in sorted_results[:3]
        self.assertTrue(set(results.keys()).issubset(words))

        results = keywords(text, top_n=3)
        assert len(results) is 3

        results = keywords(text, threshold=.1)
        for v in results.values():
            assert v >= .1

    def test_keywords_language_detect(self):
        text = "il a remporté sa première victoire dans la descente de Val Gardena en Italie"
        words = set(text.lower().split())

        results = keywords(text, language = 'detect')
        sorted_results = sorted(results.keys(), key=lambda x:results.get(x), reverse=True)
        result_keys = results.keys() if PY3 else map(lambda x: x.encode("utf-8"), results.keys())
        self.assertTrue(set(result_keys).issubset(words))

        results = keywords(text, top_n=3)
        assert len(results) is 3

        results = keywords(text, threshold=.1)
        for v in results.values():
            assert v >= .1

    def test_keywords_language(self):
        text = "il a remporté sa première victoire dans la descente de Val Gardena en Italie"
        words = set(text.lower().split())

        results = keywords(text, language = 'French')
        sorted_results = sorted(results.keys(), key=lambda x:results.get(x), reverse=True)

        result_keys = results.keys() if PY3 else map(lambda x: x.encode("utf-8"), results.keys())
        self.assertTrue(set(result_keys).issubset(words))

        results = keywords(text, top_n=3)
        assert len(results) is 3

        results = keywords(text, threshold=.1)
        for v in results.values():
            assert v >= .1

    def test_named_entities(self):
        text = "London Underground's boss Mike Brown warned that the strike ..."
        expected_entities = ("London Underground", "Mike Brown")
        expected_keys = set(["categories", "confidence"])
        entities = named_entities(text)
        for entity in expected_entities:
            assert entity in expected_entities
            assert not (set(entities[entity]) - expected_keys)

    def test_political(self):
        political_set = set(['Libertarian', 'Liberal', 'Conservative', 'Green'])
        test_string = "Guns don't kill people, people kill people."
        response = political(test_string)

        self.assertTrue(isinstance(response, dict))
        self.assertEqual(political_set, set(response.keys()))

        test_string = "pro-choice"
        response = political(test_string)

        self.assertTrue(isinstance(response, dict))
        assert response['Libertarian'] > 0.25

    def test_posneg(self):
        test_string = "Worst song ever."
        response = sentiment(test_string)

        self.assertTrue(isinstance(response, float))
        self.assertTrue(response < 0.5)

        test_string = "Best song ever."
        response = sentiment(test_string)
        self.assertTrue(isinstance(response, float))
        self.assertTrue(response > 0.5)

    def test_sentiment_hq(self):
        test_string = "Worst song ever."
        response = sentiment_hq(test_string)

        self.assertTrue(isinstance(response, float))
        self.assertTrue(response < 0.5)

        test_string = "Best song ever."
        response = sentiment_hq(test_string)
        self.assertTrue(isinstance(response, float))
        self.assertTrue(response > 0.5)

    def test_twitter_engagement(self):
        test_string = "Worst song ever."
        response = twitter_engagement(test_string)

        self.assertIsInstance(response, float)
        self.assertTrue(response <= 1)
        self.assertTrue(response >= 0)

    def test_personalities(self):
        test_string = "I love my friends!"
        response = personality(test_string)

        categories = ['extraversion', 'openness', 'agreeableness', 'conscientiousness']
        self.assertTrue(isinstance(response, dict))
        self.assertIsInstance(response['extraversion'], float)
        for category in categories:
            assert category in response.keys()

    def test_personas(self):
        test_string = "I love my friends!"
        response = personas(test_string)

        self.assertTrue(isinstance(response, dict))
        self.assertIsInstance(response["commander"], float)
        self.assertTrue(len(response.keys()))

    def test_good_fer(self):
        fer_set = set(['Angry', 'Sad', 'Neutral', 'Surprise', 'Fear', 'Happy'])
        test_face = os.path.normpath(os.path.join(DIR, "data/48by48.png"))
        response = fer(test_face)

        self.assertTrue(isinstance(response, dict))
        self.assertEqual(fer_set, set(response.keys()))

    def test_good_int_array_fer(self):
        fer_set = set(['Angry', 'Sad', 'Neutral', 'Surprise', 'Fear', 'Happy'])
        test_face = os.path.normpath(os.path.join(DIR, "data/48by48.png"))
        response = fer(test_face)

        self.assertTrue(isinstance(response, dict))
        self.assertEqual(fer_set, set(response.keys()))

    def test_happy_fer(self):
        test_face = os.path.normpath(os.path.join(DIR, "data/happy.png"))
        response = fer(test_face)
        self.assertTrue(isinstance(response, dict))
        self.assertTrue(response['Happy'] > 0.5)

    def test_happy_fer_pil(self):
        test_face = Image.open(os.path.normpath(os.path.join(DIR, "data/happy.png"))).convert('L');
        response = fer(test_face)
        self.assertTrue(isinstance(response, dict))
        self.assertTrue(response['Happy'] > 0.5)

    def test_fear_fer(self):
        test_face = os.path.normpath(os.path.join(DIR, "data/fear.png"))
        response = fer(test_face)
        self.assertTrue(isinstance(response, dict))
        self.assertTrue(response['Fear'] > 0.25)

    def test_bad_fer(self):
        fer_set = set(['Angry', 'Sad', 'Neutral', 'Surprise', 'Fear', 'Happy'])
        test_face = os.path.normpath(os.path.join(DIR, "data/64by64.png"))
        response = fer(test_face)

        self.assertTrue(isinstance(response, dict))
        self.assertEqual(fer_set, set(response.keys()))

    def test_facial_localization(self):
        test_face = os.path.normpath(os.path.join(DIR, "data/happy.png"))
        res = facial_localization(test_face)[0]
        self.assertTrue(res["top_left_corner"][0] < res["bottom_right_corner"][0])
        self.assertTrue(res["top_left_corner"][1] < res["bottom_right_corner"][1])

    def test_facial_localization_sensitivity(self):
        test_face = os.path.normpath(os.path.join(DIR, "data/happy.png"))
        low_sens = facial_localization(test_face, sensitivity=0.1)
        high_sens = facial_localization(test_face, sensitivity=0.9)
        self.assertEqual(len(low_sens), 1)
        self.assertTrue(len(high_sens) > 1)

    def test_facial_localization_crop(self):
        test_face = os.path.normpath(os.path.join(DIR, "data/happy.png"))
        res = facial_localization(test_face, crop=True)[0]
        self.assertTrue(res.get("image"))

    def test_safe_content_filtering(self):
        test_face = os.path.normpath(os.path.join(DIR, "data/happy.png"))
        response = content_filtering(test_face)
        self.assertTrue(response < 0.5)

    def test_resize_content_filtering(self):
        test_face = os.path.normpath(os.path.join(DIR, "data/happy.png"))
        response = content_filtering(test_face)
        self.assertTrue(isinstance(response, float))

    def test_good_facial_features(self):
        test_face = os.path.normpath(os.path.join(DIR, "data/48by48.png"))
        response = facial_features(test_face)

        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 48)
        self.check_range(response)

    def test_rgba_int_array_facial_features(self):
        test_face = os.path.normpath(os.path.join(DIR, "data/48by48rgba.png"))
        response = facial_features(test_face)

        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 48)
        self.check_range(response)

    def test_good_int_array_facial_features(self):
        fer_set = set(['Angry', 'Sad', 'Neutral', 'Surprise', 'Fear', 'Happy'])
        test_face = os.path.normpath(os.path.join(DIR, "data/48by48.png"))
        response = facial_features(test_face)

        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 48)
        self.check_range(response)

    def test_good_image_features_greyscale(self):
        test_image = os.path.normpath(os.path.join(DIR, "data/48by48.png"))
        response = image_features(test_image)

        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 4096)
        self.check_range(response)

    def test_good_image_features_rgb(self):
        test_image = os.path.normpath(os.path.join(DIR, "data/48by48rgb.png"))
        response = image_features(test_image)

        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 4096)
        self.check_range(response)

    def test_multi_api_image(self):
        test_data = os.path.normpath(os.path.join(DIR, "data/48by48.png"))
        response = analyze_image(test_data, apis=config.IMAGE_APIS, api_key=self.api_key)

        self.assertTrue(isinstance(response, dict))
        self.assertTrue(set(response.keys()) == set(config.IMAGE_APIS))

    def test_multi_api_text(self):
        test_data = 'clearly an english sentence'
        response = analyze_text(test_data, apis=config.TEXT_APIS, api_key=self.api_key)

        self.assertTrue(isinstance(response, dict))
        self.assertTrue(set(response.keys()) == set(config.TEXT_APIS))

    def test_intersections_not_enough_data(self):
        test_data = ['test_Data']
        self.assertRaises(
            IndicoError,
            intersections,
            test_data,
            apis=['text_tags', 'sentiment']
        )

    def test_intersections_wrong_number_of_apis(self):
        test_data = ['test data']*3
        self.assertRaises(
            IndicoError,
            intersections,
            test_data,
            apis=['text_tags', 'sentiment', 'language']
        )

    def test_intersections_bad_api_type(self):
        test_data = ['test data']*3
        self.assertRaises(
            IndicoError,
            intersections,
            test_data,
            apis=['text_tags', 'fer']
        )

    def test_intersections_valid_input(self):
        test_data = ['test data']*3
        apis = ['text_tags', 'sentiment']
        results = intersections(test_data, apis=apis)
        assert set(results.keys()) < set(apis)

    def test_intersections_valid_raw_input(self):
        test_data = {
            'sentiment': [0.1, 0.2, 0.3],
            'twitter_engagement': [0.1, 0.2, 0.3]
        }
        results = intersections(test_data, apis=['sentiment', 'twitter_engagement'])
        assert set(results.keys()) < set(test_data.keys())

    def test_language(self):
        language_set = set([
            'English',
            'Spanish',
            'Tagalog',
            'Esperanto',
            'French',
            'Chinese',
            'French',
            'Bulgarian',
            'Latin',
            'Slovak',
            'Hebrew',
            'Russian',
            'German',
            'Japanese',
            'Korean',
            'Portuguese',
            'Italian',
            'Polish',
            'Turkish',
            'Dutch',
            'Arabic',
            'Persian (Farsi)',
            'Czech',
            'Swedish',
            'Indonesian',
            'Vietnamese',
            'Romanian',
            'Greek',
            'Danish',
            'Hungarian',
            'Thai',
            'Finnish',
            'Norwegian',
            'Lithuanian'
        ])
        language_dict = language('clearly an english sentence')
        self.assertEqual(language_set, set(language_dict.keys()))
        assert language_dict['English'] > 0.25

    def test_set_cloud(self):
        test_data = 'clearly an english sentence'
        self.assertRaises(ConnectionError,
                          language,
                          test_data,
                          cloud='invalid/cloud')

        temp_cloud = config.cloud
        config.cloud = 'invalid/cloud'

        self.assertEqual(config.cloud, 'invalid/cloud')
        self.assertRaises(ConnectionError,
                          language,
                          test_data)

        config.cloud = temp_cloud

    def test_set_api_key(self):
        test_data = 'clearly an english sentence'
        self.assertRaises(IndicoError,
                          language,
                          test_data,
                          api_key ='invalid_api_key')

        temp_api_key = config.api_key
        config.api_key = 'invalid_api_key'

        self.assertEqual(config.api_key, 'invalid_api_key')
        self.assertRaises(IndicoError,
                          language,
                          test_data)

        config.api_key = temp_api_key


class NumpyImagesRun(unittest.TestCase):
    """
    Testing numpy array as images
    """
    def setUp(self):
        self.api_key = config.api_key
        try:
            import numpy as np
            globals()["np"] = np
        except ImportError:
            self.skipTest("Numpy is not installed!")

    def check_range(self, _list, minimum=0.9, maximum=0.1, span=0.5):
        vector = list(flatten(_list))
        _max = max(vector)
        _min = min(vector)
        self.assertTrue(max(vector) > maximum)
        self.assertTrue(min(vector) < minimum)
        self.assertTrue(_max - _min > span)

    def test_float_numpy_arrays(self):
        test_image = np.random.random(size=(48,48))
        response = image_features(test_image)

        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 4096)
        self.check_range(response)

    def test_float_RGB_numpy_arrays(self):
        test_image = np.random.random(size=(48,48,3))
        response = image_features(test_image)

        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 4096)
        self.check_range(response)

    def test_float_RGBA_numpy_arrays(self):
        test_image = np.random.random(size=(48,48,4))
        response = image_features(test_image)

        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 4096)
        self.check_range(response)

    def test_int_numpy_arrays(self):
        test_image = np.random.randint(0, 255, size=(48,48))
        response = image_features(test_image)

        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 4096)
        self.check_range(response)

    def test_int_RGB_numpy_arrays(self):
        test_image = np.random.randint(0, 255, size=(48,48, 3))
        response = image_features(test_image)

        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 4096)
        self.check_range(response)

    def test_int_RGBA_numpy_arrays(self):
        test_image = np.random.randint(0, 255, size=(48,48, 3))
        response = image_features(test_image)

        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 4096)
        self.check_range(response)

    def test_invalid_int_numpy_arrays(self):
        test_image = np.random.randint(255, 300, size=(48,48, 3))
        self.assertRaises(IndicoError, image_features, test_image)

    def test_invalid_int_numpy_arrays(self):
        test_image = np.random.randint(255, 300, size=(48,48, 5))
        self.assertRaises(IndicoError, image_features, test_image)

    def test_resize_content_filtering_numpy_arrays(self):
        test_image = np.random.randint(0, 255, size=(480,248, 3))
        response = content_filtering(test_image)
        self.assertTrue(isinstance(response, float))

def flatten(container):
    for i in container:
        if isinstance(i, list) or isinstance(i, tuple):
            for j in flatten(i):
                yield j
        else:
            yield i

if __name__ == "__main__":
    unittest.main()
