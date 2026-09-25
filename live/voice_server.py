"""TR-Omni-E4B voice server: the talker as a small HTTP service that streams 48 kHz speech.

For every sentence the conversation server sends the text (in its spoken form), the thinker's hidden states for its
tokens and each token's character span; the talker fuses its own token embeddings with the mapped states and
generates the speech as a continuation of the assistant's voice prompt, streaming 16-bit PCM as it is decoded.
Without states it reads the text alone (used for the short backchannels).

    python live/voice_server.py --model /path/to/TR-Omni-E4B [--port 7870]

Started by live/server.py; it runs in its own process so that the talker's and the thinker's CUDA graphs and
Python loops do not compete for one interpreter.
"""
import argparse
import asyncio
import base64
import concurrent.futures
import io
import os
import queue
import sys
import threading
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the TR-Omni-E4B folder (with talker/ and tr_omni.py)")
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--seed", type=int, default=42, help="the same noise for every sentence: a steadier voice")
    ap.add_argument("--optimize", type=int, default=1, help="torch.compile the talker (slow first sentence)")
    args = ap.parse_args()

    import soundfile as sf
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, StreamingResponse

    sys.path.insert(0, args.model)
    from tr_omni import AUDIO_START, Talker

    # cuDNN attention plans every new sequence length anew: ~0.1 s more before the first audio of each sentence of a
    # new length (i.e. almost every sentence); flash / memory-efficient attention compute the same without that cost
    torch.backends.cuda.enable_cudnn_sdp(False)

    t0 = time.time()
    tk = Talker(os.path.join(args.model, "talker"))
    # generation caches sized for a sentence after the voice prompt, not the config's default length: every
    # decode step attends over the whole static cache, so a shorter one is much faster
    tk.m.base_lm.setup_cache(1, 2048, "cuda", torch.bfloat16)
    tk.m.residual_lm.setup_cache(1, 2048, "cuda", torch.bfloat16)
    if args.optimize:
        tk.m = tk.m.optimize()
    m, cfg = tk.m, tk.cfg
    sr = int(m.sample_rate)
    # one long-lived thread runs the talker: torch.compile records its CUDA graphs per thread, so a new thread per
    # request would record them again for every sentence (~0.2 s before the first audio)
    worker = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="talker")

    def synth(text, states=None, goffs=None, cancel=None):
        """Speech for one sentence, streamed as int16 PCM chunks; stops as soon as `cancel` is set."""
        ids = tk.vtok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            return
        with torch.no_grad():
            e = m.base_lm.embed_tokens.emb.weight[torch.tensor(ids, device="cuda")].float()
            if states is not None:
                ids2, t2g = tk.align(text, goffs, states.shape[0])
                if ids2 != ids:
                    return
                seen, run, sub = {}, {}, []
                for g in t2g:
                    run[g] = run.get(g, 0) + 1
                for g in t2g:
                    sub.append(seen.get(g, 0))
                    seen[g] = seen.get(g, 0) + 1
                s = states.cuda()
                pieces = tk.mapper(s[torch.tensor(t2g, device="cuda")], torch.tensor(sub, device="cuda"),
                                   torch.tensor([run[g] for g in t2g], device="cuda")).float()
                x = tk.fusion(e, pieces)
            else:
                x = e
            P, n, pf = len(tk.prompt_ids), len(ids), tk.prompt_feat
            L = P + n + 1 + pf.shape[0]
            toks = torch.zeros(1, L, dtype=torch.long, device="cuda")
            toks[0, :P] = torch.tensor(tk.prompt_ids, device="cuda")
            toks[0, P:P + n] = torch.tensor(ids, device="cuda")
            toks[0, P + n] = AUDIO_START
            feat = torch.zeros(1, L, 4, 64, device="cuda", dtype=torch.bfloat16)
            feat[0, P + n + 1:] = pf.to("cuda", torch.bfloat16)
            am = torch.zeros(1, L, dtype=torch.int32, device="cuda")
            am[0, P + n + 1:] = 1
            mask = torch.zeros(1, L, device="cuda")
            mask[0, P:P + n] = 1
            full = torch.zeros(1, L, x.shape[-1], device="cuda")
            full[0, P:P + n] = x
        m.base_lm.embed_tokens.rep = (mask, full)
        torch.manual_seed(args.seed)
        gen = m._inference(toks, 1 - am, feat, am, min_len=2, max_len=min(n * 6 + 10, 2048),
                           inference_timesteps=cfg["inference_timesteps"], cfg_value=cfg["cfg_value"], streaming=True)
        try:
            with torch.no_grad(), m.audio_vae.streaming_decode() as dec:
                for latent, _, _ in gen:
                    if cancel is not None and cancel.is_set():
                        return
                    a = dec.decode_chunk(latent.to(torch.float32)).squeeze().float().cpu().numpy()
                    yield (np.clip(a, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        finally:
            gen.close()
            m.base_lm.embed_tokens.rep = None

    def parse(body):
        text = (body.get("text") or "").strip()
        states = goffs = None
        if body.get("states"):
            raw = np.frombuffer(base64.b64decode(body["states"]), dtype=np.int16)
            states = torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(-1, 3, 2560)
            goffs = body.get("goffs")
        return text, states, goffs

    app = FastAPI()

    @app.get("/info")
    def info():
        return {"model": "TR-Omni-E4B talker (tokens + thinker states)", "sr": sr,
                "steps": cfg["inference_timesteps"], "cfg": cfg["cfg_value"], "voice": cfg.get("voice")}

    @app.post("/tts")
    async def tts(req: Request):
        # the talker thread streams the chunks into a queue; when the client goes away (the user cut in), the
        # response's generator is closed and the cancel flag stops the talker at its next chunk
        text, states, goffs = parse(await req.json())
        q, cancel = queue.Queue(), threading.Event()

        def work():
            try:
                if not cancel.is_set():
                    for b in synth(text, states, goffs, cancel):
                        q.put(b)
            except Exception as e:
                q.put(e)
            finally:
                q.put(None)

        worker.submit(work)
        loop = asyncio.get_running_loop()

        async def body():
            try:
                while True:
                    item = await loop.run_in_executor(None, q.get)
                    if item is None:
                        return
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                cancel.set()

        return StreamingResponse(body(), media_type="application/octet-stream", headers={"X-Sample-Rate": str(sr)})

    @app.post("/tts_wav")
    async def tts_wav(req: Request):
        text, states, goffs = parse(await req.json())
        pcm = await asyncio.wrap_future(worker.submit(lambda: b"".join(synth(text, states, goffs))))
        buf = io.BytesIO()
        sf.write(buf, np.frombuffer(pcm, dtype="<i2"), sr, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav")

    # warm-up on the talker thread: the voice prompt is encoded and, with --optimize, the talker compiled and its
    # graphs recorded before anyone waits
    def warm():
        for _ in synth("Merhaba, size nasıl yardımcı olabilirim?"):
            pass

    worker.submit(warm).result()
    print(f"voice ready: {sr} Hz in {time.time() - t0:.0f}s, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB",
          flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
