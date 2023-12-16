import os
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, \
    TextIteratorStreamer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "=http://127.0.0.1:7890"
os.environ["all_proxy"] = "socks5://127.0.0.1:7890"

model = MBartForConditionalGeneration.from_pretrained(
    r"D:\zjpython_work\pyvideotrans-main\models--facebook--mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained(
    r"D:\zjpython_work\pyvideotrans-main\models--facebook--mbart-large-50-many-to-many-mmt")
# translate English to chinese
tokenizer.src_lang = "en_XX"
print('model is ok, ', model)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    messages = "".join(["".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]])  # curr_system_message +
                        for item in history_transformer_format])

    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield partial_message


if __name__ == '__main__':
    gr.ChatInterface(predict).queue().launch()
