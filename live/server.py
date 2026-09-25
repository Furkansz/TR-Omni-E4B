"""TR-Omni-E4B live conversation server: talk to the model in the browser.

    python live/server.py [--model /path/to/TR-Omni-E4B] [--port 7860]
    # then open http://localhost:7860 in Chrome and click the orb

Pipeline per user turn:
  browser VAD fragments -> Smart Turn v3.2 decides the end of the turn (short holds, the odd "hı hı" backchannel)
  -> the thinker (Gemma 4 E4B, speech-adapted) answers the audio itself, streamed token by token
  -> the reply is cut into sentences; each is normalised to its spoken form (tr_speak.py)
  -> the thinker's hidden states for it are taken in context (system prompt + the user's last 4 s + reply so far),
     on top of a per-turn KV cache of that context, on a side CUDA stream (no queueing behind the decoder)
  -> the voice server (live/voice_server.py) speaks it from tokens + states, streamed back as 48 kHz PCM
  -> the page plays it through a local WebRTC loopback so the browser's echo canceller can hear it, which lets the
     user interrupt (barge-in) without the model hearing itself.

The model code (tr_omni.py, tr_speak.py) is imported from the model folder, downloaded from the Hugging Face Hub
unless --model points to a local copy. Needs a CUDA GPU with about 17 GB free (24 GB recommended).
"""
import argparse
import asyncio
import base64
import copy
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "furkanbekmezci/TR-Omni-E4B-Turkish-Native-Speech-to-Speech-Model"
MODES = {                                   # name -> (label, instruction; None = spoken conversation)
    "sohbet": ("Sesli sohbet", None),
    "yaz": ("Yazıya dök", "Bu ses kaydını yazıya dök."),
    "ozet": ("Özetle", "Konuşmacının anlattıklarını kısaca özetle."),
    "tarz": ("Konuşma tarzı / ruh hali",
             "Konuşmacının konuşma tarzını (hız, ses tonu, tonlama, duraksamalar) ve ruh halini değerlendir."),
    "ceviri": ("İngilizceye çevir", "Söylenenleri İngilizceye çevir. Sadece çeviriyi yaz."),
    "betimle": ("Betimle", "Bu ses kaydını ayrıntılı olarak betimle: ne söyleniyor ve nasıl söyleniyor?"),
}
HISTORY_TURNS = 3              # earlier turns the thinker sees
STATE_LAYERS = (14, 28, 42)    # thinker layers the talker reads
SILENCE_PEAK = 700             # int16 peak below which a streamed chunk counts as silence (~-33 dBFS)
PIECE_GAP = 0.12               # s of silence kept between two sentences (a short breath)
# turn-taking: P(done) from Smart Turn -> answer now / after a short wait / after a longer one (maybe "hı hı")
P_NOW, P_SHORT, P_BC = 0.85, 0.50, 0.30
HOLD_SHORT, HOLD_MAX = 0.55, 1.4
MIN_FRAG = 0.5                 # s: shorter fragments are clicks and coughs
LEVEL_FLOOR = -50.0            # dBFS: quieter speech is room noise
LEVEL_DROP = 12.0              # dB: a fragment this much quieter than the open question is someone/something else
FRAG_GAP = 0.25                # s of silence put between the fragments of one turn
REOPEN_S, REOPEN_MIN = 1.5, 1.0   # speech >= 1 s right after the turn closed, before anything is heard: same turn
BC_TEXTS = ("Hı hı.", "Evet.", "Anlıyorum.", "Hı hı, evet.")
BC_MIN_SPEECH, BC_EVERY, BC_P = 4.0, 8.0, 0.5   # after >= 4 s of user speech, >= 8 s apart, half of the time
BACKCHANNELS = []              # [(sr, pcm16 bytes)] in the assistant's voice, made at startup
OM = {}                        # the model code: tr_omni and tr_speak from the model folder


def speech_level(pcm):
    """Level of a fragment's speech: mean dBFS of its louder half of 20 ms frames (pauses do not count)."""
    f = pcm[:len(pcm) // 320 * 320].reshape(-1, 320)
    e = np.sort(10 * np.log10((f.astype(np.float64) ** 2).mean(1) + 1e-10))
    return float(e[len(e) // 2:].mean())


def norm_key(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


# ------------------------------------------------------------------------------------------------------ thinker
class LiveThinker:
    """The thinker with streamed greedy decoding (one CUDA graph per token, on its own thread) and the talker's
    in-context states (a per-turn KV cache of the context, computed on a side CUDA stream)."""

    def __init__(self, path):
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        om = OM["tr_omni"]
        self.proc = AutoProcessor.from_pretrained(path)
        self.tok = self.proc.tokenizer
        model = AutoModelForMultimodalLM.from_pretrained(path, dtype=torch.bfloat16)
        model.model.vision_tower = None             # the web demo sends audio only
        model.model.embed_vision = None
        lm = model.model.language_model
        lm.embed_tokens_per_layer = om.HostEmbedding(lm.embed_tokens_per_layer)
        self.model = model.cuda().eval()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decode")   # CUDA graphs are thread-bound
        self.fast = self.pool.submit(lambda: om.FastDecoder(self.model)).result()
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.side = torch.cuda.Stream()

    def _inputs(self, msgs, prefix):
        x = self.proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                          add_generation_prompt=True)
        if prefix:                                  # every per-token tensor grows with it (mm_token_type_ids: text)
            pre = torch.tensor([self.tok(prefix, add_special_tokens=False)["input_ids"]])
            L = x["input_ids"].shape[1]
            for k in list(x.keys()):
                v = x[k]
                if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == L:
                    fill = pre if k == "input_ids" else (torch.ones_like(pre) if k == "attention_mask"
                                                          else torch.zeros_like(pre))
                    x[k] = torch.cat((v, fill.to(v.dtype)), 1)
        return x.to("cuda")

    @torch.no_grad()
    def stream(self, msgs, cancel, on_piece, max_new_tokens=1024, prefix=None):
        """Greedy reply, handed to on_piece() as it is generated. `prefix` is given as already said (the tone tag)."""
        with self.lock:
            inputs = self._inputs(msgs, prefix)

            def run():                              # always on the decode thread: its CUDA graphs live there
                ids, said = [], ""
                for t in self.fast.generate(inputs, max_new_tokens):
                    if cancel.is_set():
                        break
                    ids.append(t)
                    text = self.tok.decode(ids, skip_special_tokens=True)
                    if text.endswith("�"):     # half of a multi-byte character: wait for the rest
                        continue
                    if len(text) > len(said):
                        on_piece(text[len(said):])
                        said = text
            self.pool.submit(run).result()

    async def astream(self, msgs, cancel, max_new_tokens=1024, prefix=None):
        loop, q = asyncio.get_running_loop(), asyncio.Queue()

        def work():
            try:
                self.stream(msgs, cancel, lambda p: loop.call_soon_threadsafe(q.put_nowait, p), max_new_tokens, prefix)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            p = await q.get()
            if p is None:
                return
            if isinstance(p, Exception):
                raise p
            yield p

    @torch.no_grad()
    def states_prefix(self, pcm):
        """The context every sentence of a turn is read in (system prompt + the end of the user's audio), encoded
        once: its KV cache. states() reads each sentence on top of it."""
        om = OM["tr_omni"]
        msgs = [[{"role": "system", "content": [{"type": "text", "text": om.SYSTEM}]},
                 {"role": "user", "content": [{"type": "audio", "audio": pcm[-om.CTX_SEC * 16000:]}]}]]
        pre = self.proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                            add_generation_prompt=True)
        with self.state_lock, torch.cuda.stream(self.side):
            ids = pre["input_ids"].cuda()
            extra = {k: v.cuda() for k, v in pre.items() if k not in ("input_ids", "attention_mask")}
            out = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, **extra)
            self.side.synchronize()
        return out.past_key_values

    @torch.no_grad()
    def states(self, prefix_cache, before, text):
        """Hidden states (layers 14/28/42) for `text`, spoken after `before`, as the talker was trained on them:
        '[ton: nötr] <reply so far> <sentence>' after the cached context -> {"states": b64 bf16, "goffs": spans}."""
        head = f"{OM['tr_omni'].TONE} " + (before + " " if before else "")
        enc = self.tok(head + text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        first = next(j for j, (a, b) in enumerate(offs) if b > len(head))
        with self.state_lock, torch.cuda.stream(self.side):
            ids = torch.tensor([enc["input_ids"]], device="cuda")
            cache = copy.deepcopy(prefix_cache)     # the context stays intact for the next sentence
            out = self.model.model.language_model(input_ids=ids, past_key_values=cache, use_cache=True,
                                                  output_hidden_states=True)
            n = ids.shape[1]
            hs = torch.stack([out.hidden_states[i][0, first:n] for i in STATE_LAYERS], 1).to(torch.bfloat16).cpu()
        H = len(head)
        return {"states": base64.b64encode(hs.contiguous().view(torch.int16).numpy().tobytes()).decode(),
                "goffs": [[max(a - H, 0), max(b - H, 0)] for a, b in offs[first:]]}


def build_messages(history, audio, instruction):
    om = OM["tr_omni"]
    msgs = [] if instruction else [{"role": "system", "content": [{"type": "text", "text": om.SYSTEM}]}]
    for h in history[-HISTORY_TURNS:]:
        msgs.append({"role": "user", "content": [{"type": "audio", "audio": s} for s in om.split_long(h["audio"])]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": h["reply"]}]})
    content = [{"type": "text", "text": instruction}] if instruction else []
    content += [{"type": "audio", "audio": s} for s in om.split_long(audio)]
    msgs.append({"role": "user", "content": content})
    return msgs


class Segmenter:
    """Cuts the streamed reply into speakable pieces at natural boundaries: whole sentences, clauses once a sentence
    runs long. The first piece ends at the first sentence end or at a comma after a few words, so the voice starts
    early without clipped two-word fragments."""
    END = re.compile(r"(?:(?<!\d)\.|[\!\?…])+[\"'”’\)]*\s")    # "13. yüzyıl": a dot after a number is no end
    MID = re.compile(r"[,;:]\s")

    def __init__(self):
        self.buf, self.first = "", True

    def push(self, text):
        self.buf += text
        out = []
        while True:
            cut = self._cut()
            if cut is None:
                return out
            piece, self.buf = self.buf[:cut].strip(), self.buf[cut:]
            if piece:
                out.append(piece)
                self.first = False

    def _cut(self):
        for m in self.END.finditer(self.buf):
            if m.group(0).rstrip("\"'”’) \n\t") == "." and not OM["tr_speak"].dot_ends_sentence(self.buf, m.start()):
                continue                            # "II. Mehmed", "Dr. Ali", "M.Ö. 3000"
            if len(self.buf[:m.end()].split()) >= 2:
                return m.end()
        words = len(self.buf.split())
        if words >= (4 if self.first else 18):
            for m in (self.MID.finditer(self.buf) if self.first else reversed(list(self.MID.finditer(self.buf)))):
                if len(self.buf[:m.end()].split()) >= (3 if self.first else 8):
                    return m.end()
        if words >= (22 if self.first else 30):     # no punctuation: cut at a word boundary
            return self.buf.rstrip().rfind(" ") + 1 or None
        return None

    def flush(self):
        piece, self.buf = self.buf.strip(), ""
        return [piece] if piece else []


def voice_stream(url, text, cancel, states=None):
    """(sample rate, pcm16 bytes) chunks from the voice server."""
    body = {"text": text}
    if states is not None:
        body.update(states)
    with requests.post(f"{url}/tts", json=body, stream=True, timeout=(5, 120)) as r:
        r.raise_for_status()
        sr = int(r.headers.get("X-Sample-Rate", "48000"))
        for chunk in r.iter_content(chunk_size=None):
            if cancel.is_set():
                return
            if chunk:
                yield sr, chunk


# ---------------------------------------------------------------------------------------------------- one turn
class Session:
    def __init__(self):
        self.history, self.mode, self.task, self.cancel, self.turn = [], "sohbet", None, threading.Event(), 0
        self.frags, self.hold, self.last_bc, self.bc_i = [], None, 0.0, 0
        self.speech_since_bc, self.turn_t0, self.turn_audio, self.heard = 0.0, 0.0, None, False
        self.level = None              # the user's speech level (dBFS), learnt from accepted fragments
        self.heard_n = None            # barge-in: sentences of the reply heard before the user cut in

    def cancel_hold(self):
        if self.hold and not self.hold.done():
            self.hold.cancel()
        self.hold = None

    def merged(self):
        gap = np.zeros(int(16000 * FRAG_GAP), np.float32)
        out = []
        for f in self.frags:
            out += [f, gap]
        return np.concatenate(out[:-1]) if out else np.zeros(0, np.float32)

    async def stop_turn(self):
        self.cancel.set()
        if self.task and not self.task.done():
            try:
                await asyncio.wait_for(self.task, 10)
            except (asyncio.TimeoutError, Exception):
                self.task.cancel()
        self.task = None


async def run_turn(ws, sess, thinker, voice_url, pcm, cancel):
    loop = asyncio.get_running_loop()
    t0 = time.time()
    sess.turn += 1
    turn, stats = sess.turn, {}
    await ws.send_json({"type": "turn_start", "turn": turn, "dur": round(len(pcm) / 16000, 1)})
    instruction = MODES.get(sess.mode, MODES["sohbet"])[1]
    msgs = build_messages(sess.history if instruction is None else [], pcm, instruction)
    sentences, said = asyncio.Queue(), []       # said: sentences handed to the voice
    prefix_fut = loop.run_in_executor(None, thinker.states_prefix, pcm)   # alongside the reply's first tokens
    progress = {"t": time.time()}

    async def speak():
        spoken = set()
        while True:
            s = await sentences.get()
            if s is None or cancel.is_set():
                return
            key = norm_key(s)
            if not key or key in spoken:       # a looping reply must not be read out twice
                continue
            spoken.add(key)
            stats.setdefault("t_sentence", round(time.time() - t0, 2))
            text = OM["tr_speak"].speakable(s)
            if not re.search(r"\w", text):
                continue
            try:
                prefix = await prefix_fut
                states = await loop.run_in_executor(None, thinker.states, prefix, " ".join(said), text)
            except Exception as e:
                await ws.send_json({"type": "error", "error": f"thinker states: {e!r}"[:300]})
                continue
            stats.setdefault("t_states", round(time.time() - t0, 2))
            said.append(text)
            chunks = asyncio.Queue()

            def fetch(text=text, states=states):
                try:
                    for item in voice_stream(voice_url, text, cancel, states):
                        loop.call_soon_threadsafe(chunks.put_nowait, item)
                except Exception as e:
                    loop.call_soon_threadsafe(chunks.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(chunks.put_nowait, None)

            threading.Thread(target=fetch, daemon=True).start()
            rest, caption, lead, held = b"", s, True, []
            while True:
                item = await chunks.get()
                if item is None:
                    if held and not cancel.is_set():        # keep ~0.12 s of the sentence's trailing silence
                        await ws.send_bytes(b"".join(held)[:int(PIECE_GAP * stats.get("sr", 48000)) * 2])
                    break
                if isinstance(item, Exception):
                    await ws.send_json({"type": "error", "error": f"voice: {item!r}"[:300]})
                    break
                if cancel.is_set():
                    continue
                sr, b = item
                if "sr" not in stats:
                    stats["sr"] = sr
                    await ws.send_json({"type": "audio_meta", "turn": turn, "sr": sr})
                stats.setdefault("t_audio", round(time.time() - t0, 2))
                sess.heard = True
                progress["t"] = time.time()
                b = rest + b
                cut = len(b) - len(b) % 2
                rest = b[cut:]
                if not cut:
                    continue
                b = b[:cut]
                quiet = np.abs(np.frombuffer(b, dtype="<i2")).max(initial=0) < SILENCE_PEAK
                if quiet and lead:                  # a later sentence: drop the silence it starts with
                    continue
                lead = False
                if quiet:                           # maybe the sentence's trailing silence: hold it back
                    held.append(b)
                    continue
                if held:                            # it was a pause inside the sentence: send it after all
                    await ws.send_bytes(b"".join(held))
                    held = []
                if caption is not None:             # the sentence's text, shown when its sound starts
                    await ws.send_json({"type": "say", "turn": turn, "text": caption})
                    caption = None
                await ws.send_bytes(b)

    speaker = asyncio.create_task(speak())
    seg, reply = Segmenter(), ""
    prefix = OM["tr_omni"].TONE if instruction is None else None   # the tone tag is given, not generated
    try:
        async for p in thinker.astream(msgs, cancel, prefix=prefix):
            progress["t"] = time.time()
            stats.setdefault("t_text", round(time.time() - t0, 2))
            if not reply:
                p = p.lstrip()
            reply += p
            if cancel.is_set():
                break
            await ws.send_json({"type": "text", "turn": turn, "delta": p})
            for s in seg.push(p):
                await sentences.put(s)
    except Exception as e:
        await ws.send_json({"type": "error", "error": f"model: {e!r}"[:300]})
    for s in seg.flush():
        await sentences.put(s)
    await sentences.put(None)
    await speaker
    if not cancel.is_set():
        sess.history.append({"audio": pcm, "reply": reply.strip()})
    elif said:                                  # interrupted: keep what the user heard (or was being said)
        n = len(said) if sess.heard_n is None else min(sess.heard_n, len(said))
        if n:
            sess.history.append({"audio": pcm, "reply": " ".join(said[:n]) + " …", "cut": True})
    sess.history = sess.history[-HISTORY_TURNS:]
    await ws.send_json({"type": "turn_done", "turn": turn, "cancelled": cancel.is_set(),
                        "t_text": stats.get("t_text"), "t_audio": stats.get("t_audio"),
                        "t_total": round(time.time() - t0, 2)})
    print(f"[turn {turn}] {len(pcm) / 16000:.1f}s in, first text {stats.get('t_text')}s, first sentence "
          f"{stats.get('t_sentence')}s, states {stats.get('t_states')}s, first audio {stats.get('t_audio')}s, "
          f"{len(reply)} chars{' (interrupted)' if cancel.is_set() else ''}", flush=True)


# ------------------------------------------------------------------------------------------------ turn-taking
class SmartTurn:
    """Pipecat Smart Turn v3.2 (BSD-2, int8 ONNX on CPU): P(the user's turn is complete) from the last 8 s."""

    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor
        path = hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx")
        so = ort.SessionOptions()
        so.inter_op_num_threads, so.intra_op_num_threads = 1, 2
        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self.fe = WhisperFeatureExtractor(chunk_length=8)

    def __call__(self, pcm):
        n = 8 * 16000
        x = pcm[-n:].astype(np.float32)
        if len(x) < n:
            x = np.concatenate((np.zeros(n - len(x), np.float32), x))
        f = self.fe(x, sampling_rate=16000, return_tensors="np", do_normalize=True).input_features.astype(np.float32)
        return float(np.asarray(self.sess.run(None, {"input_features": f})[0]).reshape(-1)[0])


SMART_TURN = {"model": None}


async def turn_score(pcm):
    st = SMART_TURN["model"]
    if st is None:                              # no Smart Turn: every pause the browser detects ends the turn
        return 1.0
    return await asyncio.get_running_loop().run_in_executor(None, st, pcm)


def make_backchannels(voice_url):
    """Short acknowledgements in the assistant's voice, made once."""
    import io
    import soundfile as sf
    for t in BC_TEXTS:
        try:
            r = requests.post(f"{voice_url}/tts_wav", json={"text": t}, timeout=120)
            a, sr = sf.read(io.BytesIO(r.content), dtype="int16")
            if 0.25 <= len(a) / sr <= 1.5:
                BACKCHANNELS.append((sr, a.tobytes()))
        except Exception as e:
            print(f"backchannel {t!r}: {e!r}", flush=True)


def make_app(thinker, voice_url):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=os.path.join(HERE, "web", "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = open(os.path.join(HERE, "web", "index.html"), encoding="utf-8").read()
        return page.replace("%MODES%", json.dumps({k: v[0] for k, v in MODES.items()}, ensure_ascii=False))

    @app.get("/api/info")
    def info():
        try:
            voice = requests.get(f"{voice_url}/info", timeout=2).json()
        except Exception as e:
            voice = {"error": repr(e)[:200]}
        return {"thinker": "TR-Omni-E4B thinker", "voice": voice}

    def start_turn(ws, sess, whole):
        sess.frags = []
        sess.turn_t0, sess.turn_audio, sess.heard, sess.heard_n = time.time(), whole, False, None
        sess.cancel = threading.Event()
        sess.task = asyncio.create_task(run_turn(ws, sess, thinker, voice_url, whole, sess.cancel))

    async def hold_then_answer(ws, sess, wait):
        try:
            await asyncio.sleep(wait)
        except asyncio.CancelledError:
            return
        if sess.frags and (sess.task is None or sess.task.done()):
            start_turn(ws, sess, sess.merged())

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        sess = Session()
        try:
            while True:
                m = await ws.receive()
                if m["type"] == "websocket.disconnect":
                    break
                if m.get("bytes"):
                    pcm = np.frombuffer(m["bytes"], dtype="<i2").astype(np.float32) / 32768.0
                    if len(pcm) < 16000 * MIN_FRAG:
                        continue
                    lvl = speech_level(pcm)
                    busy = sess.task is not None and not sess.task.done()
                    joining = bool(sess.frags) or busy
                    if lvl < LEVEL_FLOOR or (joining and sess.level is not None and lvl < sess.level - LEVEL_DROP):
                        continue                            # room noise, or someone else in the room
                    sess.level = max(sess.level, lvl) if (sess.frags and sess.level is not None) else lvl
                    if busy:
                        # a reply is being prepared: the question is never replaced; only a real continuation
                        # (>= 1 s of speech, right after the turn closed, nothing heard yet) joins it
                        reopen = (sess.turn_audio is not None and not sess.heard
                                  and time.time() - sess.turn_t0 < REOPEN_S and len(pcm) >= 16000 * REOPEN_MIN)
                        if not reopen:
                            continue
                        await sess.stop_turn()
                        sess.frags = [sess.turn_audio]
                        if sess.history and sess.history[-1].get("cut"):
                            sess.history.pop()
                    sess.cancel_hold()
                    sess.frags.append(pcm)
                    sess.speech_since_bc += len(pcm) / 16000
                    whole = sess.merged()
                    p = await turn_score(whole) if len(whole) < 16000 * 25 else 1.0
                    if p >= P_NOW:
                        start_turn(ws, sess, whole)
                    else:
                        await ws.send_json({"type": "hold"})
                        if (p < P_BC and BACKCHANNELS and sess.speech_since_bc >= BC_MIN_SPEECH
                                and time.time() - sess.last_bc >= BC_EVERY and random.random() < BC_P):
                            sr, pcm16 = BACKCHANNELS[sess.bc_i % len(BACKCHANNELS)]
                            sess.bc_i += 1
                            sess.last_bc, sess.speech_since_bc = time.time(), 0.0
                            await ws.send_json({"type": "bc", "sr": sr, "pcm": base64.b64encode(pcm16).decode(),
                                                "dur": len(pcm16) / 2 / sr})
                        sess.hold = asyncio.create_task(
                            hold_then_answer(ws, sess, HOLD_SHORT if p >= P_SHORT else HOLD_MAX))
                elif m.get("text"):
                    c = json.loads(m["text"])
                    if c.get("type") == "speech_start":
                        sess.cancel_hold()                  # the user goes on: keep collecting
                    elif c.get("type") == "cancel":
                        await sess.stop_turn()
                    elif c.get("type") == "barge":          # the user cut in while the voice played
                        sess.heard_n = max(0, int(c.get("heard") or 0))
                        await sess.stop_turn()
                        sess.cancel_hold()
                    elif c.get("type") == "reset":
                        await sess.stop_turn()
                        sess.cancel_hold()
                        sess.frags = []
                        sess.history.clear()
                    elif c.get("type") == "mode":
                        sess.mode = c.get("mode") if c.get("mode") in MODES else "sohbet"
        except WebSocketDisconnect:
            pass
        finally:
            sess.cancel_hold()
            await sess.stop_turn()

    return app


# ------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=None, help="local TR-Omni-E4B folder (default: download from the Hub)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--voice-port", type=int, default=7870)
    ap.add_argument("--voice-url", default=None, help="use an already running voice server instead of starting one")
    ap.add_argument("--no-smart-turn", action="store_true", help="end the turn at every pause the browser detects")
    args = ap.parse_args()
    model = args.model
    if model is None:
        from huggingface_hub import snapshot_download
        model = snapshot_download(REPO_ID)
    sys.path.insert(0, model)
    import tr_omni
    import tr_speak
    OM.update(tr_omni=tr_omni, tr_speak=tr_speak)
    torch.set_num_threads(4)
    # cuDNN attention plans every new sequence length anew (~0.05-0.1 s per sentence, and every sentence is new);
    # flash / memory-efficient attention compute the same without that cost
    torch.backends.cuda.enable_cudnn_sdp(False)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # stop cleanly, voice server included

    voice_url, voice = args.voice_url, None
    if voice_url is None:                       # the talker in its own process
        voice_url = f"http://127.0.0.1:{args.voice_port}"
        voice = subprocess.Popen([sys.executable, os.path.join(HERE, "voice_server.py"), "--model", model,
                                  "--port", str(args.voice_port)],
                                 env={**os.environ, "TQDM_DISABLE": "1"})   # no progress bar per sentence
    try:
        run(args, model, voice, voice_url)
    finally:                                    # the voice server never outlives this one
        if voice is not None:
            voice.terminate()
            try:
                voice.wait(10)
            except subprocess.TimeoutExpired:
                voice.kill()


def run(args, model, voice, voice_url):
    t0 = time.time()
    thinker = LiveThinker(os.path.join(model, "thinker"))

    async def serve():
        import uvicorn
        # the first multimodal request pays one-off costs and captures the CUDA graphs: before anyone waits
        warm = (np.random.default_rng(0).standard_normal(16000 * 2) * 0.01).astype(np.float32)
        _ = [p async for p in thinker.astream(build_messages([], warm, None), threading.Event())]
        loop = asyncio.get_running_loop()           # and the talker-state path once (its first call is slow)
        cache = await loop.run_in_executor(None, thinker.states_prefix, warm)
        await loop.run_in_executor(None, thinker.states, cache, "", "Merhaba.")
        print(f"thinker ready in {time.time() - t0:.0f}s", flush=True)
        for _ in range(240):                    # the voice compiles its graphs on start (a few minutes at most)
            try:
                requests.get(f"{voice_url}/info", timeout=2).raise_for_status()
                break
            except Exception:
                if voice is not None and voice.poll() is not None:
                    raise SystemExit("the voice server stopped; see its log above")
                await asyncio.sleep(5)
        await asyncio.get_running_loop().run_in_executor(None, make_backchannels, voice_url)
        if not args.no_smart_turn:
            try:
                SMART_TURN["model"] = SmartTurn()
            except Exception as e:
                print(f"Smart Turn unavailable ({e!r}): every pause ends the turn", flush=True)
        print(f"ready: open http://{args.host}:{args.port} in Chrome", flush=True)
        config = uvicorn.Config(make_app(thinker, voice_url), host=args.host, port=args.port, log_level="warning",
                                ws_max_size=64 * 2**20, timeout_graceful_shutdown=3)
        await uvicorn.Server(config).serve()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
