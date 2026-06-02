"""Measure the two cloud legs of the voice pipeline from THIS machine, over the
same network path the live Pipecat demo used.

  • Anthropic LLM leg  = time-to-first-token (TTFT) for the demo model.
  • Deepgram STT leg   = streaming *finalization* latency: stream a wav at
                         real-time pace, send Finalize, time until the final
                         transcript returns. This is the STT network+processing
                         cost, separate from the Silero VAD endpointing wait
                         (that stays its own leg — we don't double-count it).

Keys come from .env via os.environ and are never printed.
Run:  .venv/bin/python measure_legs.py [wav]
"""
import os, sys, json, time, wave, asyncio, statistics
from dotenv import load_dotenv

load_dotenv()
DG_KEY = os.environ["DEEPGRAM_API_KEY"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
DG_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-2")
WAV = sys.argv[1] if len(sys.argv) > 1 else "client_out.wav"


async def measure_anthropic(n=6):
    """TTFT over n runs (median + range)."""
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=ANTHROPIC_KEY)
    ttfts = []
    for i in range(n):
        t0 = time.perf_counter()
        async with client.messages.stream(
            model=ANTHROPIC_MODEL, max_tokens=64,
            system="You are a friendly voice assistant. Reply in ONE short sentence.",
            messages=[{"role": "user", "content": "Hey, what can you help me with?"}],
        ) as stream:
            async for _text in stream.text_stream:
                ttfts.append((time.perf_counter() - t0) * 1000.0)
                break  # first token only
        await asyncio.sleep(0.3)  # don't hammer the API
    return ttfts


async def measure_deepgram_once():
    """Stream the wav real-time, Finalize, time until the final transcript."""
    import websockets
    w = wave.open(WAV)
    sr, ch = w.getframerate(), w.getnchannels()
    pcm = w.readframes(w.getnframes())
    w.close()
    url = (f"wss://api.deepgram.com/v1/listen?model={DG_MODEL}"
           f"&encoding=linear16&sample_rate={sr}&channels={ch}"
           f"&interim_results=true&punctuate=true")
    state = {}
    async with websockets.connect(
        url, additional_headers={"Authorization": f"Token {DG_KEY}"}, max_size=None
    ) as ws:
        async def sender():
            # Send faster-than-real-time in ~100 ms blocks (Deepgram buffers and
            # transcribes ahead fine); this avoids the idle-timeout the slow
            # real-time pacing was hitting. We then Finalize and time the flush —
            # i.e. the STT network+processing finalization cost.
            step = int(sr * ch * 2 / 1000) * 100  # 100 ms of audio per block
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step])
                await asyncio.sleep(0.005)
            state["t_finalize"] = time.perf_counter()
            await ws.send(json.dumps({"type": "Finalize"}))

        async def receiver():
            async for msg in ws:
                d = json.loads(msg)
                if d.get("type") != "Results":
                    continue
                txt = d["channel"]["alternatives"][0].get("transcript", "")
                # the first is_final AFTER we sent Finalize = finalization latency
                if d.get("is_final") and txt and "t_finalize" in state:
                    state["t_final"] = time.perf_counter()
                    state["txt"] = txt
                    return

        await asyncio.gather(sender(), receiver())
        await ws.send(json.dumps({"type": "CloseStream"}))
    return (state["t_final"] - state["t_finalize"]) * 1000.0, state.get("txt", "")


def fmt(xs):
    return (f"median {statistics.median(xs):6.1f} ms   "
            f"(min {min(xs):.0f}  max {max(xs):.0f}  n={len(xs)})")


async def main():
    print(f"network path = this machine → cloud (same as the live demo)\n")

    print(f"[LLM]  Anthropic TTFT  model={ANTHROPIC_MODEL}")
    ttfts = await measure_anthropic()
    print(f"       {fmt(ttfts)}\n")

    print(f"[STT]  Deepgram finalization  model={DG_MODEL}  wav={WAV}")
    finals = []
    for i in range(4):
        ms, txt = await measure_deepgram_once()
        finals.append(ms)
        await asyncio.sleep(0.3)
    print(f"       {fmt(finals)}")
    print(f"       (last transcript: {txt!r})")


if __name__ == "__main__":
    asyncio.run(main())
