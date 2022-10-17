import time

from keybert import KeyBERT


class KeyWordExtractor:
    """Extracts Key word from text"""

    def __init__(self, model='all-MiniLM-L6-v2',top_n=5):

        self.kw_model = KeyBERT(model=model)
        self.top_n = top_n

    def predict(self, doc):
        t1 = time.time()
        keywords = self.kw_model.extract_keywords(docs=doc,
                                                  top_n=self.top_n,
                                                  stop_words='english',
                                                  keyphrase_ngram_range=(1, 2),
                                                  use_mmr=True, diversity=0.7)

        latency = time.time()-t1
        return keywords


