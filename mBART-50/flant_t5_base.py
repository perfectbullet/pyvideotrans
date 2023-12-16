from transformers import T5Tokenizer, T5ForConditionalGeneration

import os

# 为模型下载设置代理
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "=http://127.0.0.1:7890"
os.environ["all_proxy"] = "socks5://127.0.0.1:7890"

# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", proxies={'http': '127.0.0.1:7890', 'https': '127.0.0.1:7890', 'socks': '127.0.0.1:7890'})
tokenizer = T5Tokenizer.from_pretrained(r"D:\zjpython_work\google-flan-t5-large-model")
model = T5ForConditionalGeneration.from_pretrained(r"D:\zjpython_work\google-flan-t5-large-model")
#
# input_text = "translate English to Chinese: How old are you?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# outputs = model.generate(input_ids)
# result = tokenizer.decode(outputs[0])
# print(result)


# Tokenize and convert to tensor
english_text = 'How old are you?'
input_ids = tokenizer.encode("translate English to Chinese: " + english_text, return_tensors="pt")

# Generate translation
output = model.generate(input_ids)

# Decode and print the translation
chinese_translation = tokenizer.decode(output[0], encoding='utf-8')
print(chinese_translation)
