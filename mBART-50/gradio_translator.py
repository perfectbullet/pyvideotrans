import gradio as gr
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

print('https://www.gradio.app/docs/blocks')
# english_translator = gr.load(name="spaces/gradio/english_translator")
# english_generator = pipeline("text-generation", model="distilgpt2")
#
#
# def generate_text(text):
#     english_text = english_generator(text)[0]["generated_text"]
#     german_text = english_translator(english_text)
#     return english_text, german_text


title = "网信平台组AI小组翻译玩具"

description = "MBart的Gradio demo. 使用它, 只需添加您的文本, 或单击一个例子, 然后提交翻译."

examples = [
    ["Beijing is the capital of China", ]
]

model = MBartForConditionalGeneration.from_pretrained(
    r"D:\zjpython_work\pyvideotrans-main\models--facebook--mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained(
    r"D:\zjpython_work\pyvideotrans-main\models--facebook--mbart-large-50-many-to-many-mmt")
# translate English to chinese
tokenizer.src_lang = "en_XX"
print('model is ok, ', model)


def generate_text(text):
    encoded_ar = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_ar,
        forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
    )
    res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return res.pop()


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                seed = gr.Text(label="Input Phrase")
            with gr.Column():
                english = gr.Text(label="Generated English Text")
                # german = gr.Text(label="Generated German Text")
        btn = gr.Button("Generate")
        btn.click(generate_text, inputs=[seed], outputs=[english, ])
        gr.Examples(["My name is Clara and I am"], inputs=[seed])
    demo.launch()
