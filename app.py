""" Extract key words from patent abstracts """
import os
import time
from typing import Callable
from PIL.Image import Image
from pathlib import Path
import gradio as gr
import logging
import pandas as pd
from keywod_extractor import KeyWordExtractor
from google_patent_scraper import scraper_class

os.environ["CUDA_VISIBLE_DEVICES"] = ""
logging.basicConfig(level=logging.INFO)
APP_DIR = Path(__file__).resolve().parent  # what is the directory for this application?
README = os.path.join(APP_DIR, "README.md")
EG_PATH = os.path.join(APP_DIR, "examples")


def main():
    # close all previous sessions
    gr.close_all()
    extractor = ExtractorBackend()
    demo = make_demo(fn=extractor.run)
    # frontend = make_frontend(
    #     fn=extractor.run
    # )
    # frontend.launch(
    #     server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
    #     server_port=8000,  # set a port to bind to, failing if unavailable
    #     # share=True,  # should we create a (temporary) public link on https://gradio.app?
    #     # favicon_path=FAVICON,  # what icon should we display in the address bar?
    #
    # )
    demo.launch(share=False, debug=True)


def get_patent_abstract(patent_number):
    scraper = scraper_class(return_abstract=True)
    err_1, soup_1, url_1 = scraper.request_single_patent(patent_number)
    patent_1_parsed = scraper.get_scraped_data(soup_1, patent_number, url_1)
    abstract = patent_1_parsed['abstract_text']
    return abstract, gr.Button.update(visible=True), gr.Button.update(visible=False)


def abstract_textbox_appear(choice):
    if choice:
        return gr.components.Textbox(visible=True)
    else:
        return gr.components.Textbox(visible=False)


def change_textbox(choice):
    if choice == "Abstract":
        return [gr.Textbox.update(visible=True),
                gr.Textbox.update(visible=False, value=""),
                gr.Textbox.update(visible=False, value=""),
                gr.Button.update(visible=True),
                gr.Button.update(visible=False)]
    elif choice == 'US Patent Number':
        return [gr.Textbox.update(visible=False, value=""),
                gr.Textbox.update(visible=True),
                gr.Textbox.update(visible=True),
                gr.Button.update(visible=False),
                gr.Button.update(visible=True)]


def make_demo(fn: Callable[[str], list]):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                radio = gr.Radio(
                    ["Abstract", "US Patent Number"], label="Select to enter either patent number or abstract"
                )
                abstract1 = gr.Textbox(label='Abstract', lines=8, visible=False, interactive=True,
                                       placeholder="Enter patent abstract here")
                patent_number = gr.Textbox(label='US Patent Number', lines=1, visible=False, interactive=True,
                                           placeholder="e.g. US10188981B2")
                abstract2 = gr.Textbox(label='Abstract', visible=False, interactive=True,
                                       placeholder="click Get Abstract to populate this field")
                b1 = gr.Button('Extract Keywords', visible=False)
                b2 = gr.Button('Get Abstract', visible=False)
            with gr.Column():
                keyword_df = gr.Dataframe()
                keyword_text = gr.Textbox(lines=8)
                highlt_keyword_text = gr.HighlightedText()

        gr.Examples(
            examples=[_abstract_example()],
            inputs=abstract1, label='Abstract Example'
            # outputs=,
            # fn=mirror
        )

        gr.Examples(
            examples=["US10188981B2"],
            inputs=patent_number, label="US Patent Number Example"
            # outputs=,
            # fn=mirror
        )
        radio.change(fn=change_textbox, inputs=radio, outputs=[abstract1, patent_number, abstract2, b1, b2])

        b2.click(get_patent_abstract, inputs=patent_number, outputs=[abstract2, b1, b2])
        b1.click(fn, inputs=[abstract1, abstract2], outputs=[keyword_df, keyword_text, highlt_keyword_text])

    return demo


def make_frontend(fn: Callable[[str], list]):
    allow_flagging = "never"
    readme = _load_readme(with_logging=allow_flagging == "manual")
    outputs = [gr.components.Dataframe(),
               gr.components.Textbox(visible=True)]
    frontend = gr.Interface(
        fn=fn,  # which Python function are we interacting with?
        outputs=outputs,  # what output widgets does it need? the default text widget
        # what input widgets does it need? we configure an image widget
        inputs=[gr.components.Textbox(placeholder=_place_holder_input_text()),
                gr.components.Checkbox(label="US Patent Number?")],
        title="📝 Key Word Extraction",  # what should we display at the top of the page?

        description=__doc__,  # what should we display just above the interface?
        article=readme,  # what long-form content should we display below the interface?
        examples=[[_abstract_example(), False], ["US10188981B2", True]],  # which potential inputs should we provide?
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
    def __init__(self, model):
        if model:
            model = KeyWordExtractor(model=model)
        else:
            model = KeyWordExtractor()
        self._predict = model.predict

    def run(self, doc1, doc2):
        abstract = doc1 + doc2

        preds, metrics = self._extract_with_metrics(abstract)
        df = pd.DataFrame(list(zip(preds, metrics)),
                          columns=['Key Word', 'Relevance'])
        kw_in_abstract = self.key_word_in_abstract(preds, abstract)

        return df, abstract, kw_in_abstract

    def key_word_in_abstract(self, preds, abstract):
        kw_in_abstract = []
        import re
        #
        # text = "Hello, this red car is very beautiful and nice. Also, this car is green."
        # words_list = ["car", "red", "green"]

        for pred in preds:
            pred_dict = dict()
            pred_dict['entity'] = 'KW'
            pred_dict['score'] = 0.99
            for m in re.finditer(pred, abstract):
                pred_dict['index'] = None
                pred_dict['start'] = m.start()
                pred_dict['end'] = m.end()
                kw_in_abstract.append(pred_dict)
                print(m.group(), m.start(), m.end())

        return kw_in_abstract

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
