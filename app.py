""" extract key words from patents"""
import os
import time
from typing import Callable
from PIL.Image import Image
from pathlib import Path
import gradio as gr
import logging
import pandas as pd
from keywod_extractor import KeyWordExtractor

os.environ["CUDA_VISIBLE_DEVICES"] = ""
logging.basicConfig(level=logging.INFO)
APP_DIR = Path(__file__).resolve().parent  # what is the directory for this application?
README = os.path.join(APP_DIR, "README.md")
EG_PATH = os.path.join(APP_DIR, "examples")


def main():
    # close all previous sessions
    gr.close_all()
    extractor = ExtractorBackend()
    frontend = make_frontend(
        fn=extractor.run
    )
    frontend.launch(
        server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
        server_port=8000,  # set a port to bind to, failing if unavailable
        # share=True,  # should we create a (temporary) public link on https://gradio.app?
        # favicon_path=FAVICON,  # what icon should we display in the address bar?
    )


def get_patent_abstract():
    text = 'Test Abstract'

    return text


place_holder_text = 'A method for preparing a polygraphene membrane includes adding ' \
                    'graphite and sodium nitrate into sulfuric acid to form a first mixture; ' \
                    'adding potassium permanganate solution into the first mixture to form a ' \
                    'second mixture;adding hydrogen peroxide solution to the second mixture to ' \
                    'form a mixture including soluble manganese ions; filtering the mixture including ' \
                    'soluble manganese ions to form an aqueous suspension; centrifuging the aqueous suspension; ' \
                    'performing ultrasonication of the suspension to obtain graphene oxide sheets; ' \
                    'acylating the graphene oxide sheets to prepare an acylated graphene oxide sheet;' \
                    'and polymerizing the acylated graphene oxide sheets to prepare polygraphene.'


def make_frontend(fn: Callable[[str], list]):
    allow_flagging = "never"
    readme = _load_readme(with_logging=allow_flagging == "manual")
    outputs = [gr.components.Dataframe()]
    frontend = gr.Interface(
        fn=fn,  # which Python function are we interacting with?
        outputs=outputs,  # what output widgets does it need? the default text widget
        # what input widgets does it need? we configure an image widget
        inputs=gr.components.Textbox(placeholder=_place_holder_input_text()),
        title="📝 Key Word Extraction",  # what should we display at the top of the page?

        description=__doc__,  # what should we display just above the interface?
        article=readme,  # what long-form content should we display below the interface?
        examples=[place_holder_text],  # which potential inputs should we provide?
        cache_examples=False,  # should we cache those inputs for faster inference? slows down start
        # allow_flagging=allow_flagging,  # should we show users the option to "flag" outputs?
    )

    return frontend


def _place_holder_input_text():
    place_holder_text = 'A method for preparing a polygraphene membrane includes adding ' \
                        'graphite and sodium nitrate into sulfuric acid to form a first mixture; ' \
                        'adding potassium permanganate solution into the first mixture to form a ' \
                        'second mixture;adding hydrogen peroxide solution to the second mixture to ' \
                        'form a mixture including soluble manganese ions; filtering the mixture including ' \
                        'soluble manganese ions to form an aqueous suspension; centrifuging the aqueous suspension; ' \
                        'performing ultrasonication of the suspension to obtain graphene oxide sheets; ' \
                        'acylating the graphene oxide sheets to prepare an acylated graphene oxide sheet;' \
                        ' and polymerizing the acylated graphene oxide sheets to prepare polygraphene.'
    return place_holder_text


def _load_readme(with_logging=False):
    with open(README) as f:
        lines = f.readlines()
        # if not with_logging:
        #     lines = lines[:lines.index("<!-- logging content below -->\n")]

        readme = "".join(lines)
    return readme


def _abstract_example():
    return _place_holder_input_text()


def _examples():
    examples = os.listdir(EG_PATH)
    examples = [os.path.join(EG_PATH, i) for i in examples]
    return examples


class ExtractorBackend:
    def __init__(self):
        model = KeyWordExtractor()
        self._predict = model.predict

    def run(self, doc):
        preds, metrics = self._extract_with_metrics(doc)
        df = pd.DataFrame(list(zip(preds, metrics)),
                          columns=['Key Word', 'Relevance'])

        return df

    def _extract_with_metrics(self, doc):
        keywords = list()
        relevances = list()
        t1 = time.time()
        preds = self._predict(doc)
        exec_time = time.time() - t1
        for word, metric in preds:
            keywords.append(word)
            relevances.append(metric)
        return keywords, relevances


def _get_ouput_components():
    outputs = list()
    return outputs


if __name__ == "__main__":
    # parser = _make_parser()
    # args = parser.parse_args()
    main()