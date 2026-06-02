"""Local Pipecat voice agent:  mic -> Deepgram STT -> Anthropic LLM ->
megakernel TTS (remote, over WebSocket) -> speakers.

Run on YOUR machine (it needs the mic/speakers). The megakernel TTS runs on the
RTX 5090 box and is reached via an SSH tunnel:
    ssh -N -L 8765:localhost:8765 -p <port> root@<host>

Env (.env):  DEEPGRAM_API_KEY, ANTHROPIC_API_KEY, optional ANTHROPIC_MODEL, TTS_WS_URL
"""
import os, sys, json, asyncio, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # quiet pipecat 1.3 API-transition notices
import websockets
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    TTSAudioRawFrame, ErrorFrame, LLMRunFrame, TranscriptionFrame,
    UserStartedSpeakingFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

load_dotenv()
logger.remove()
logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))  # quiet the DEBUG firehose
TTS_WS_URL = os.environ.get("TTS_WS_URL", "ws://localhost:8765")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
TTS_SR = 24000  # Qwen3-TTS codec sample rate


class MegakernelTTSService(TTSService):
    """Streams PCM from the remote megakernel TTS server, frame by frame."""

    def __init__(self, url=TTS_WS_URL, speaker="Ryan", language="English", **kwargs):
        super().__init__(sample_rate=TTS_SR,
                         settings=TTSSettings(model="qwen3-tts-12hz-0.6b-megakernel",
                                              voice=speaker, language=language),
                         **kwargs)
        self._url, self._speaker, self._language, self._ws = url, speaker, language, None

    async def _conn(self):
        ws = self._ws
        if ws is not None and getattr(ws, "close_code", None) is not None:
            ws = None  # connection closed; reconnect
        if ws is None:
            # ping_interval=None: don't let keepalive pings time out over the
            # high-latency tunnel during long bot-speaking gaps (that was closing
            # the connection mid-conversation).
            ws = await websockets.connect(self._url, max_size=None,
                                          ping_interval=None, close_timeout=5)
            self._ws = ws
        return ws

    async def run_tts(self, text, context_id):
        try:
            ws = await self._conn()
            await self.start_ttfb_metrics()
            await ws.send(json.dumps({"text": text, "speaker": self._speaker,
                                      "language": self._language, "chunk_frames": 3}))
            await self.start_tts_usage_metrics(text)
            first = True
            while True:
                msg = await ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    if first:
                        await self.stop_ttfb_metrics()
                        first = False
                    yield TTSAudioRawFrame(audio=bytes(msg), sample_rate=self.sample_rate,
                                           num_channels=1, context_id=context_id)
                else:
                    evt = json.loads(msg)
                    if evt.get("event") in ("done", "error"):
                        break
        except Exception as e:
            try:
                if self._ws is not None:
                    await self._ws.close()
            except Exception:
                pass
            self._ws = None  # force a clean reconnect on the next turn
            logger.warning(f"TTS hiccup (reconnecting next turn): {type(e).__name__}: {e}")
            yield ErrorFrame(error=f"megakernel tts: {e}")


class TurnLogger(FrameProcessor):
    """Clean INFO-level turn indicators (who's speaking) + your transcript."""
    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            logger.info(f"🗣️  YOU: {frame.text}")
        elif isinstance(frame, UserStartedSpeakingFrame):
            logger.info("🎤 listening… (you're speaking)")
        elif isinstance(frame, BotStartedSpeakingFrame):
            logger.info("🔊 BOT speaking…")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.info("   …bot finished")
        await self.push_frame(frame, direction)


async def main():
    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ))
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = AnthropicLLMService(api_key=os.environ["ANTHROPIC_API_KEY"], model=ANTHROPIC_MODEL)
    tts = MegakernelTTSService()

    context = LLMContext(messages=[{
        "role": "system",
        "content": ("You are a friendly voice assistant running on an RTX 5090, "
                    "speaking through a custom CUDA megakernel. Reply in ONE short, "
                    "natural sentence — never more than one sentence. Start by "
                    "greeting the user in one short sentence."),
    }])
    agg = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        TurnLogger(),
        agg.user(),
        llm,
        tts,
        transport.output(),
        agg.assistant(),
    ])
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Greet first (LocalAudioTransport has no client-connect event): queue an
    # LLM run so the assistant speaks its opening line, then the user replies.
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=True)
    logger.info(f"Voice agent up. TTS={TTS_WS_URL} model={ANTHROPIC_MODEL}. Speak into your mic.")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
