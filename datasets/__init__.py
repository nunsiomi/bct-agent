"""datasets/ is data-only.

Code that used to live here (``utils.py``, ``validation.py``) moved to the
``core`` package. Thin shims remain in this directory so legacy import paths
still resolve, but new code should import from ``core`` directly.

Layout of this directory:
- ``persona_signals.csv``        Yelp / Amazon / Goodreads signals (built by the data-prep notebook)
- ``jumia_reviews_*.csv``        Scraped Jumia product reviews (Nigerian ground truth)
- ``utils.py``                   Deprecated re-export -> core.*
- ``validation.py``              Deprecated re-export -> core.validation
"""
