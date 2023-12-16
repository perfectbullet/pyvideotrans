from transformers import T5ForConditionalGeneration, T5Tokenizer

import os

# 为模型下载设置代理
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "=http://127.0.0.1:7890"
os.environ["all_proxy"] = "socks5://127.0.0.1:7890"

# Load pre-trained T5 model and tokenizer
model_name = "t5-large"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# English text to be translated
english_text = "Hello, how are you?"

# Tokenize and convert to tensor
input_ids = tokenizer.encode("translate English to Chinese: " + english_text, return_tensors="pt")

# Generate translation
output = model.generate(input_ids)

# Decode and print the translation
chinese_translation = tokenizer.decode(output[0], skip_special_tokens=True)
print(chinese_translation)