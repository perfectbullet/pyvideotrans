# load  microsoft/speecht5_tts
import os
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "=http://127.0.0.1:7890"
os.environ["all_proxy"] = "socks5://127.0.0.1:7890"

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")


from datasets import load_dataset, Audio

dataset = load_dataset(
    "facebook/voxpopuli", "nl", split="train"
)

tokenizer = processor.tokenizer

def extract_all_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


from collections import defaultdict
speaker_counts = defaultdict(int)

for speaker_id in dataset["speaker_id"]:
    speaker_counts[speaker_id] += 1


import matplotlib.pyplot as plt

plt.figure()
plt.hist(speaker_counts.values(), bins=20)
plt.ylabel("Speakers")
plt.xlabel("Examples")
plt.show()
