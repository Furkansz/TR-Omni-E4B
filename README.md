# TR-Omni-E4B — live speech-to-speech demo

Real-time voice conversation in the browser with **[TR-Omni-E4B](https://huggingface.co/furkanbekmezci/TR-Omni-E4B-Turkish-Native-Speech-to-Speech-Model)**,
a two-component (thinker–talker) Turkish speech-to-speech model. The model weights, the model card and a minimal
offline inference script are on the Hugging Face Hub; this repository contains the live, streaming application.

> **Türkçe —** Bu depo, TR-Omni-E4B ile tarayıcıdan canlı sesli sohbet için gereken uygulamayı içerir: cümle cümle
> akışlı konuşma, Smart Turn ile konuşma sırası algılama, kısa geri bildirimler ("hı hı") ve yankı engellemeli söz kesme.
> Model ağırlıkları Hugging Face'tedir; uygulama ilk çalıştırmada modeli otomatik indirir.

## Features

- **Sentence-level streaming.** The thinker writes the reply token by token; each sentence is voiced as soon as it
  is complete. On one 2g MIG slice of an H200 the first audio arrives about 0.7 s after the end of the user's turn
  is detected (median over 20 spoken questions, server side; the page itself waits for 350 ms of silence).
- **Turn-taking.** The browser segments speech with a VAD; [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3)
  decides whether the user has finished, with graded waiting times and occasional short backchannels.
- **Barge-in.** The user can interrupt at any time. The assistant's voice is played through a local WebRTC loopback
  into an `<audio>` element, so Chrome's echo canceller uses it as its reference and the model does not hear itself.
  The voice ducks when speech starts and stops after a sustained stretch of speech; short acknowledgements and
  noise let it continue. The conversation history keeps only the sentences the user heard.
- **Fast talker conditioning.** The thinker's hidden states for each sentence are computed on top of a per-turn KV
  cache of the context, on a separate CUDA stream, so they do not queue behind the decoder.
- **Turkish reading rules.** Numbers, ordinals (also Roman), dates, units, abbreviations and formulas are spoken
  correctly (`tr_speak.py` from the model repository).

## Requirements

- Linux with an NVIDIA GPU and about 17 GB of free GPU memory (24 GB recommended)
- Python 3.10–3.12, CUDA-enabled PyTorch 2.8 or newer
- Chrome (or another Chromium browser) for the web page; a microphone and speakers or headphones

## Quickstart

```bash
git clone https://github.com/Furkansz/TR-Omni-E4B.git
cd TR-Omni-E4B
pip install -r requirements.txt
python live/server.py                          # downloads the model (~21 GB) on the first run
# or: python live/server.py --model /path/to/TR-Omni-E4B
```

Open **http://localhost:7860** and click the orb. The first start compiles CUDA graphs and takes a few minutes. On a
remote GPU machine, forward the port first: `ssh -L 7860:localhost:7860 user@gpu-host`.

## How it works

```
browser ── VAD fragments ──► live/server.py ── Smart Turn v3.2 ──► end of turn?
                                  │
                                  ├─ thinker (Gemma 4 E4B, speech-adapted): streamed reply, one CUDA graph per token
                                  ├─ per sentence: spoken form (tr_speak) → hidden states in context
                                  │   (cached context, side CUDA stream)
                                  ▼
                          live/voice_server.py: talker (VoxCPM2 / Trendyol-TTS) → 48 kHz PCM, streamed
                                  │
browser ◄── audio chunks + captions ──┘   (played through a WebRTC loopback: echo cancellation, barge-in)
```

The voice server runs in its own process, started by `live/server.py`. Both servers listen on `127.0.0.1` only.

## Performance notes

- The talker runs on one long-lived thread: `torch.compile` records its CUDA graphs per thread, so a thread per
  request would record them again for every sentence (about 0.2 s).
- cuDNN attention is disabled in both processes. It builds a new plan for every new sequence length, which means
  almost every sentence (about 0.1 s each); flash and memory-efficient attention compute the same without that cost.
- Together these took the median first-audio latency on the 20-question set from 0.93 s to 0.73 s, with the same
  replies and no loss in intelligibility (Whisper WER of the spoken replies 3.7 % → 2.8 %, within sampling noise).

## Tuning

- Barge-in: `BARGE` in `live/web/index.html` (speech probability 0.85, duck after 200 ms, stop after 650 ms).
- End of speech in the browser: `redemptionMs` (350 ms) in `live/web/index.html`.
- Turn-taking: `P_NOW`, `P_SHORT`, `HOLD_SHORT`, `HOLD_MAX` and the backchannel settings in `live/server.py`.

## Limitations

- One conversation at a time per server (one GPU).
- Echo cancellation and barge-in are tuned for Chrome; other browsers may behave differently. Headphones avoid echo
  altogether.
- The model's own limitations (knowledge, a single synthetic voice) are described in the
  [model card](https://huggingface.co/furkanbekmezci/TR-Omni-E4B-Turkish-Native-Speech-to-Speech-Model).

## Responsible use

The assistant voice is synthetic. Do not use the model to impersonate real people, for fraud or for disinformation,
and label generated speech as AI-generated.

## License

Apache-2.0 (`LICENSE`). Third-party components loaded at runtime: Smart Turn v3 (BSD-2-Clause, downloaded from the
Hugging Face Hub), [@ricky0123/vad-web](https://github.com/ricky0123/vad) (ISC) and onnxruntime-web (MIT), both from
jsDelivr. See the model card for the licenses of the model and its training data.
