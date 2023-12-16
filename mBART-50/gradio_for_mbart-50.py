import gradio as gr

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

title = "网信平台组AI小组翻译玩具"

description = "MBart的Gradio demo. 使用它, 只需添加您的文本, 或单击一个例子, 然后提交翻译."

examples = [
    ["Beijing is the capital of China", ]
]

model = MBartForConditionalGeneration.from_pretrained(r"D:\zjpython_work\pyvideotrans-main\models--facebook--mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained(r"D:\zjpython_work\pyvideotrans-main\models--facebook--mbart-large-50-many-to-many-mmt")
# translate English to chinese
tokenizer.src_lang = "en_XX"
print('model is ok, ', model)


def inference(text):
    encoded_ar = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_ar,
        forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
    )
    res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return res.pop()


def greet(name):
    return "Hello " + name + "!"


def demo():
    gr.Interface(
        fn=inference,
        inputs=gr.Textbox(label="请输入翻译文本", placeholder="翻译文本"),
        outputs=gr.Textbox(label="翻译结果"),
        examples=examples,
        title=title,
        description=description,
        api_name='translate'

    ).launch()


def demo2():
    with gr.Blocks() as demo:
        # name = gr.Textbox(label="请输入翻译文本", placeholder="翻译文本")
        # output = gr.Textbox(label="翻译结果")
        gr.Interface(fn=inference, inputs=gr.Textbox(label="请输入翻译文本", placeholder="翻译文本"), outputs=gr.Textbox(label="翻译结果"), examples=examples, title=title, description=description)
        commit_btn = gr.Button("提交")
        commit_btn.click(fn=inference)

    demo.launch()


def demo3():
    '''
     gradio-3.50.2
     gradio-client-0.6.1
     版本
    '''
    gr.Interface(
        fn=inference,
        inputs=gr.Textbox(label="请输入翻译文本", placeholder="翻译文本"),
        outputs=gr.Textbox(label="翻译结果"),
        examples=examples,
        title=title,
        description=description,
        api_name='translate'

    ).launch()


if __name__ == '__main__':
    demo3()
