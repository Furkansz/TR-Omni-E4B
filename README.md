# TR-Omni-E4B live demo

This repository contains the browser interface and streaming server for [TR-Omni-E4B](https://huggingface.co/furkanbekmezci/TR-Omni-E4B-Turkish-Native-Speech-to-Speech-Model), a Turkish native speech-to-speech model. The model weights, model card and offline inference code are on Hugging Face.

The GitHub page is the source repository. Start the server on a machine with an NVIDIA GPU, then open its local address in your browser. You can use the same computer for both, or connect your laptop to a remote GPU over SSH.

> **Türkçe:** Bu depo tarayıcıdaki canlı sohbet uygulamasının kodunu içerir. Model Hugging Face'ten indirilir; sunucuyu çalıştırdıktan sonra arayüze tarayıcınızda `http://localhost:7860` adresinden girersiniz.

## Requirements

- Linux with an NVIDIA GPU; about 17 GB of free GPU memory is needed, and 24 GB gives more headroom.
- Python 3.10–3.12 with a CUDA-enabled PyTorch installation (2.8 or newer).
- Chrome or another Chromium browser, plus a microphone. Headphones help avoid echo.
- Space for the model download: about 21 GB on first run.

## Run it

```bash
git clone https://github.com/Furkansz/TR-Omni-E4B.git
cd TR-Omni-E4B
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python live/server.py
```

Wait for the `ready` message, then open **http://localhost:7860** in Chrome and allow microphone access. Click the orb to start or stop listening. The first launch downloads the model and compiles CUDA graphs, so it takes a few minutes. If the model is already on disk, pass its directory instead:

```bash
python live/server.py --model /path/to/TR-Omni-E4B
```

### GPU on another machine

Run `python live/server.py` on the GPU server. On the computer with your microphone, forward the port over SSH:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@gpu-server
```

Keep that SSH connection open and visit **http://localhost:7860** on your computer. The address refers to the forwarded server, not to a website hosted by GitHub. Browsers allow microphone access on `localhost`; for access through a public domain, serve the app over HTTPS with WebSocket support. The server has no authentication, so do not expose it publicly without adding access control.

## How the live app works

The browser detects speech and sends audio to `live/server.py`. Smart Turn v3.2 decides whether a pause ends the user's turn. The thinker answers the audio directly; as it finishes each sentence, the server prepares its spoken form and extracts the thinker's contextual hidden states. `live/voice_server.py` feeds those states and the response tokens to the talker, then streams 48 kHz audio back to the browser.

The browser plays audio through a local WebRTC loopback so Chrome's echo canceller can use it as a reference. This enables barge-in: when the user starts speaking, playback ducks and a sustained interruption stops the response. Short pauses in a longer user turn can trigger a brief backchannel before the assistant answers.

In a 20-question test on an H200 2g MIG slice, the median time from detected end of turn to first audio was **0.73 s**, measured on the server. The browser's 350 ms silence window, network delay and playback are outside that measurement. Expect different latency on other hardware.

## Configuration and limits

- `python live/server.py --help` lists server options, including `--model`, `--host`, `--port` and `--no-smart-turn`.
- Turn-taking thresholds are near the top of [`live/server.py`](live/server.py); the browser's silence and barge-in settings are in [`live/web/index.html`](live/web/index.html).
- The server handles one conversation at a time. Echo cancellation and barge-in are tuned for Chrome; other browsers may behave differently.
- The application sends microphone audio to the GPU server you run. It does not send audio to a third-party speech API. The browser loads VAD and ONNX runtime assets from jsDelivr; the model and Smart Turn weights are downloaded from Hugging Face on first use.

For model evaluation, limitations and voice-use guidance, see the [model card](https://huggingface.co/furkanbekmezci/TR-Omni-E4B-Turkish-Native-Speech-to-Speech-Model).

## License

Apache-2.0; see [`LICENSE`](LICENSE). Runtime components include [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) (BSD-2-Clause), [@ricky0123/vad-web](https://github.com/ricky0123/vad) (ISC) and onnxruntime-web (MIT). The model card covers the model's components and training data.
