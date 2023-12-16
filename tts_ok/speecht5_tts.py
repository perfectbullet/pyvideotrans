import os
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "=http://127.0.0.1:7890"
os.environ["all_proxy"] = "socks5://127.0.0.1:7890"

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
print('synthesiser ok, ', synthesiser)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", "default", split="validation")
# embeddings_dataset = load_dataset("hanamizuki-ai/genshin-voice-v3.4-mandarin", "zh-CN", split="validation")
print('hanamizuki-ai/genshin-voice-v3.4-mandarin ok, ', embeddings_dataset)
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
print('hanamizuki-ai/genshin-voice-v3.4-mandarin ok, ', embeddings_dataset)
# You can replace this embedding with your own as well.


if __name__ == '__main__':
    # pip install --upgrade pip
    # pip install --upgrade transformers sentencepiece datasets[audio]
    # https://huggingface.co/microsoft/speecht5_tts
    #
    #
    # D:\zjpython_work\zj_envs\pyvideotrans-main\Scripts\python.exe D:/zjpython_work/pyvideotrans-main/tts_ok/speecht5_tts.py
    # config.json: 100%|██████████| 2.06k/2.06k [00:00<?, ?B/s]
    # D:\zjpython_work\zj_envs\pyvideotrans-main\lib\site-packages\huggingface_hub\file_download.py:147: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\gx\.cache\huggingface\hub. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
    # To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to see activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
    #   warnings.warn(message)
    # pytorch_model.bin: 100%|██████████| 585M/585M [01:42<00:00, 5.72MB/s]
    # tokenizer_config.json: 100%|██████████| 232/232 [00:00<00:00, 116kB/s]
    # spm_char.model: 100%|██████████| 238k/238k [00:00<00:00, 408kB/s]
    # added_tokens.json: 100%|██████████| 40.0/40.0 [00:00<?, ?B/s]
    # special_tokens_map.json: 100%|██████████| 234/234 [00:00<?, ?B/s]
    # preprocessor_config.json: 100%|██████████| 433/433 [00:00<?, ?B/s]
    # config.json: 100%|██████████| 636/636 [00:00<?, ?B/s]
    # pytorch_model.bin: 100%|██████████| 50.7M/50.7M [00:08<00:00, 5.71MB/s]
    # Downloading builder script: 100%|██████████| 1.36k/1.36k [00:00<?, ?B/s]
    # Downloading readme: 100%|██████████| 1.01k/1.01k [00:00<?, ?B/s]
    # Downloading data: 100%|██████████| 17.9M/17.9M [00:03<00:00, 5.56MB/s]
    # Generating validation split: 7931 examples [00:10, 771.04 examples/s]
    #
    # Process finished with exit code 0D:\zjpython_work\zj_envs\pyvideotrans-main\Scripts\python.exe D:/zjpython_work/pyvideotrans-main/tts_ok/speecht5_tts.py
    # config.json: 100%|██████████| 2.06k/2.06k [00:00<?, ?B/s]
    # D:\zjpython_work\zj_envs\pyvideotrans-main\lib\site-packages\huggingface_hub\file_download.py:147: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\gx\.cache\huggingface\hub. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
    # To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to see activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
    #   warnings.warn(message)
    # pytorch_model.bin: 100%|██████████| 585M/585M [01:42<00:00, 5.72MB/s]
    # tokenizer_config.json: 100%|██████████| 232/232 [00:00<00:00, 116kB/s]
    # spm_char.model: 100%|██████████| 238k/238k [00:00<00:00, 408kB/s]
    # added_tokens.json: 100%|██████████| 40.0/40.0 [00:00<?, ?B/s]
    # special_tokens_map.json: 100%|██████████| 234/234 [00:00<?, ?B/s]
    # preprocessor_config.json: 100%|██████████| 433/433 [00:00<?, ?B/s]
    # config.json: 100%|██████████| 636/636 [00:00<?, ?B/s]
    # pytorch_model.bin: 100%|██████████| 50.7M/50.7M [00:08<00:00, 5.71MB/s]
    # Downloading builder script: 100%|██████████| 1.36k/1.36k [00:00<?, ?B/s]
    # Downloading readme: 100%|██████████| 1.01k/1.01k [00:00<?, ?B/s]
    # Downloading data: 100%|██████████| 17.9M/17.9M [00:03<00:00, 5.56MB/s]
    # Generating validation split: 7931 examples [00:10, 771.04 examples/s]
    #
    # Process finished with exit code 0
    speech = synthesiser("Process finished with exit code 0D",
                         forward_params={"speaker_embeddings": speaker_embedding})

    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
