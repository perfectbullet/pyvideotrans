
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained(r"D:\zjpython_work\pyvideotrans-main\t5_model")
model = T5ForConditionalGeneration.from_pretrained(r"D:\zjpython_work\pyvideotrans-main\t5_model")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
