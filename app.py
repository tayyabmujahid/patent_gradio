""" extract key words from patents"""
import os
from typing import Callable
from PIL.Image import Image
from pathlib import Path
import gradio as gr
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = ""
logging.basicConfig(level=logging.INFO)
APP_DIR = Path(__file__).resolve().parent  # what is the directory for this application?
README = os.path.join(APP_DIR, "README.md")
EG_PATH = os.path.join(APP_DIR, "examples")


def main():
    frontend = make_frontend(fn=get_patent_abstract)
    frontend.launch(
        server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
        server_port=8000,  # set a port to bind to, failing if unavailable
        # share=True,  # should we create a (temporary) public link on https://gradio.app?
        # favicon_path=FAVICON,  # what icon should we display in the address bar?
    )


def get_patent_abstract():
    text = 'Test Abstract'
    return text


def make_frontend(fn: Callable[[Image], str]):
    allow_flagging = "never"
    readme = _load_readme(with_logging=allow_flagging == "manual")
    frontend = gr.Interface(
        fn=fn,  # which Python function are we interacting with?
        outputs=gr.components.Textbox(),  # what output widgets does it need? the default text widget
        # what input widgets does it need? we configure an image widget
        inputs=gr.components.Textbox(),
        title="📝 Key Word Extraction",  # what should we display at the top of the page?

        description=__doc__,  # what should we display just above the interface?
        article=readme,  # what long-form content should we display below the interface?
        examples=_examples(),  # which potential inputs should we provide?
        cache_examples=False,  # should we cache those inputs for faster inference? slows down start
        # allow_flagging=allow_flagging,  # should we show users the option to "flag" outputs?
    )

    return frontend
def _load_readme(with_logging=False):
    with open(README) as f:
        lines = f.readlines()
        # if not with_logging:
        #     lines = lines[:lines.index("<!-- logging content below -->\n")]

        readme = "".join(lines)
    return readme


def _examples():
    examples = os.listdir(EG_PATH)
    print(examples)

    examples = [os.path.join(EG_PATH, i) for i in examples]
    return examples


if __name__ == "__main__":
    # parser = _make_parser()
    # args = parser.parse_args()
    main()
