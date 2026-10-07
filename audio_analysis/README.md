# Audio analysis with EmbeddingGemma 2

`audio_embed.py` turns a folder of recordings into a searchable, classifiable
index using Google's **EmbeddingGemma 2** (`google/embeddinggemma-2`), which
embeds audio and text into one 768-d space. Only the text + audio encoders are
loaded (570M params), so it runs on a laptop CPU or in Termux on a phone.

## Setup

```bash
# ffmpeg must be on PATH (apt install ffmpeg | brew install ffmpeg | pkg install ffmpeg)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# first run downloads ~1.5 GB of weights from Hugging Face (needs `huggingface-cli login`
# if the repo asks you to accept the Gemma terms)
```

## Commands

```bash
python audio_embed.py index ~/recordings --index ./idx            # embed everything (incremental)
python audio_embed.py search "dog barking" "someone laughing" --index ./idx -k 10
python audio_embed.py search "doorbell" --chunks --index ./idx    # rank 60 s windows, with timestamps
python audio_embed.py similar ~/recordings/clip.m4a --index ./idx # audio -> audio
python audio_embed.py classify --labels "speech,music,traffic,silence,birdsong" --index ./idx
python audio_embed.py cluster -n 8 --describe "speech,music,nature,machinery" --index ./idx
python audio_embed.py duplicates --threshold 0.95 --index ./idx
python audio_embed.py info --index ./idx
```

Every command accepts `--json` for machine-readable output.

## How it works

* ffmpeg decodes any format (wav, mp3, m4a, opus, video containers) to 16 kHz mono float32.
* Files are split into windows (default 60 s, `--window`, max 300 s; the model's
  8192-token context fits about 327 s of audio at 25 tokens/s).
* Each window is embedded with `model.encode({"audio": {"array": wave, "sampling_rate": 16000}})`.
  A file-level vector is the normalised mean of its windows.
* Text queries use the `SearchQuery` task prompt (asymmetric text-to-audio retrieval).
  Audio inputs take no prompt, per the model card.
* Embeddings are L2-normalised; scores are cosine similarities.
* `--dim 256` at index time uses Matryoshka truncation (3x smaller index, ~95% quality on audio).
  Queries automatically use the same dimension.
* Index = `index.npz` (vectors) + `meta.json` (files, windows, timestamps). Re-running
  `index` only embeds new or changed files.

## Notes

* Use `float32` on CPU and `bfloat16` on GPU. Never float16: the model returns NaNs.
* For Android/Termux: `pkg install python ffmpeg`, then the same pip install. CPU
  float32 at 570M params needs roughly 2.5 GB RAM; expect a few seconds per 60 s window.
* `AUDIO_EMBED_FAKE=1` swaps in a deterministic fake encoder so you can test the
  plumbing without downloading weights.
