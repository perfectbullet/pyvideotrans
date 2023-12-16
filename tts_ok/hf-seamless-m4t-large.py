import os
from transformers import AutoProcessor, SeamlessM4TModel

os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "=http://127.0.0.1:7890"
os.environ["all_proxy"] = "socks5://127.0.0.1:7890"

processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-large")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-large")
