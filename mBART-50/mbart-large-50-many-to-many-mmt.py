import os

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# 为模型下载设置代理

os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "=http://127.0.0.1:7890"
os.environ["all_proxy"] = "socks5://127.0.0.1:7890"

model = MBartForConditionalGeneration.from_pretrained(r"D:\zjpython_work\pyvideotrans-main\models--facebook--mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained(r"D:\zjpython_work\pyvideotrans-main\models--facebook--mbart-large-50-many-to-many-mmt")

text = '''How can I fine-tune mBART-50 for machine translation in the transformers Python library so that it learns a new word?'''
# translate chinese to English
tokenizer.src_lang = "en_XX"
encoded_ar = tokenizer(text, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_ar,
    forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
)
res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(res)
# => ['最后,我使用Twine进行上传。 devpi接口脚本("devpi")很有趣,但我认为我们不希望它安装在所有我需要它。谢谢。']
