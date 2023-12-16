from transformers import T5Tokenizer, T5ForConditionalGeneration

# tokenizer = T5Tokenizer.from_pretrained(r"D:\zjpython_work\pyvideotrans-main\t5_small_model")
# model = T5ForConditionalGeneration.from_pretrained(r"D:\zjpython_work\pyvideotrans-main\t5_small_model")
#
# from transformers import T5Tokenizer, T5ForConditionalGeneration
#
# tokenizer = T5Tokenizer.from_pretrained(r"D:\zjpython_work\pyvideotrans-main\t5_small_model")
# model = T5ForConditionalGeneration.from_pretrained(r"D:\zjpython_work\pyvideotrans-main\t5_small_model")
from pathlib import Path

cache_dir = str(Path(cache_dir).expanduser().resolve())
