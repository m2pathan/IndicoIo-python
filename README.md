indicoio-python
===============

A wrapper for the [indico API](http://indico.io).

The indico API is free to use, and no training data is required.  


Installation
------------
From PyPI:
```bash
pip install indicoio
```

From source:
```bash
git clone https://github.com/IndicoDataSolutions/IndicoIo-python.git
python setup.py install
```

API Keys + Setup
----------------
For API key registration and setup, checkout our [quickstart guide](http://docs.indico.io/v2.0/docs/api-keys).

Full Documentation
------------
Detailed documentation and further code examples are available at [indico.io/docs](https://indico.io/docs).

Supported APIs:
------------

- Positive/Negative Sentiment Analysis
- Political Sentiment Analysis
- Image Feature Extraction
- Facial Emotion Recognition
- Facial Feature Extraction
- Language Detection
- Text Topic Tagging

Examples
--------
```python
>>> from indicoio import political, sentiment, language, text_tags, keywords, fer, facial_features, image_features

>>> indicoio.config.api_key = "YOUR_API_KEY"

>>> political("Guns don't kill people. People kill people.")
{u'Libertarian': 0.47740164630834825, u'Green': 0.08454409540443657, u'Liberal': 0.16617097211030055, u'Conservative': 0.2718832861769146}

>>> sentiment('Worst movie ever.')
0.07062467665597527

>>> sentiment('Really enjoyed the movie.')
0.8105182526856075

>>> text_tags("Facebook blog posts about Android tech make better journalism than most news outlets.")

>>> text_tags(test_text, threshold=0.1) # return only keys with value > 0.1
{u'startups_and_entrepreneurship': 0.21888586688354486}

>>> text_tags(test_text, top_n=1) # return only keys with top_n values
{u'startups_and_entrepreneurship': 0.21888586688354486}

>>> import numpy as np

>>> test_face = np.linspace(0,50,48*48).reshape(48,48)

>>> fer(test_face)
{u'Angry': 0.08843749137458341, u'Sad': 0.39091163159204684, u'Neutral': 0.1947947999669361, u'Surprise': 0.03443785859010413, u'Fear': 0.17574534848440568, u'Happy': 0.11567286999192382}

>>> facial_features(test_face)
[0.0, -0.02568680526917187, 0.21645604230056517, ..., 3.0342637531932777]

>>> language('Quis custodiet ipsos custodes')
{u'Swedish': 0.00033330636691921914, u'Lithuanian': 0.007328693814717631, u'Vietnamese': 0.0002686116137658802, u'Romanian': 8.133913804076592e-06, ...}

>>> keywords("Facebook blog posts about Android tech make better journalism than most news outlets.", top_n=3)
{u'android': 0.10602030910588661,
 u'journalism': 0.13466866170166855,
 u'outlets': 0.13930405357808642}
```

Batch API
---------
Each `indicoio` function can process many examples with a single request. Simply pass in a list of inputs and receive a list of results in return.
```python
>>> from indicoio import sentiment

>>> sentiment(['Best day ever', 'Worst day ever'])
[0.9899001220871786, 0.005709885173415242]
```


Calling multiple APIs with a single function
---------
There are two multiple API functions `predict_text` and `predict_image`. These functions are similar to the existing api functions, but take in an additional `apis` argument as a list of strings of API names (defaults to all existing apis). `predict_text` accepts a list of existing text APIs and vice versa for `predict_image`. These functions also support batch as the other functions do.

Accepted text API names: `text_tags, political, sentiment, language`

Accepted image API names: `fer, facial_features, image_features`

```python
>>> from indicoio import predict_text, predict_image, predict_text, predict_image

>>> predict_text('Best day ever', apis=["sentiment", "language"])
{'sentiment': 0.9899001220871786, 'language': {u'Swedish': 0.0022464881013042294, u'Vietnamese': 9.887170914498351e-05, ...}}

>>> predict_text(['Best day ever', 'Worst day ever'], apis=["sentiment", "language"])
{'sentiment': [0.9899001220871786, 0.005709885173415242], 'language': [{u'Swedish': 0.0022464881013042294, u'Vietnamese': 9.887170914498351e-05, u'Romanian': 0.00010661175919993216, ...}, {u'Swedish': 0.4924352805804646, u'Vietnamese': 0.028574824174911372, u'Romanian': 0.004185623723173551, u'Dutch': 0.000717033819689362, u'Korean': 0.0030093489153785826, ...}]}

>>> import numpy as np

>>> test_face = np.linspace(0,50,48*48).reshape(48,48).tolist()

>>> predict_image(test_face, apis=["fer", "facial_features"])
{'facial_features': [0.0, -0.026176479280200796, 0.20707644777495776, ...], 'fer': {u'Angry': 0.08877494466353497, u'Sad': 0.3933999409104264, u'Neutral': 0.1910612654566151, u'Surprise': 0.0346146405941845, u'Fear': 0.17682159820518667, u'Happy': 0.11532761017005204}}

>>> predict_image([test_face, test_face], apis=["fer", "facial_features"])
{'facial_features': [[0.0, -0.026176479280200796, 0.20707644777495776, ...], [0.0, -0.026176479280200796, 0.20707644777495776, ...]], 'fer': [{u'Angry': 0.08877494466353497, u'Sad': 0.3933999409104264, u'Neutral': 0.1910612654566151, u'Surprise': 0.0346146405941845, u'Fear': 0.17682159820518667, u'Happy': 0.11532761017005204}, { u'Angry': 0.08877494466353497, u'Sad': 0.3933999409104264, u'Neutral': 0.1910612654566151, u'Surprise': 0.0346146405941845, u'Fear': 0.17682159820518667, u'Happy': 0.11532761017005204}]}
```
