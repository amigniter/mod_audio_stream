from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Optional
from .audio import b64encode_pcm16

@dataclass(frozen=True)
class FsAudioContract:
    sample_rate: int
    channels: int
    frame_ms: int


def fs_stream_audio_json(
    frame_pcm16: bytes,
    contract: FsAudioContract,
    *,
    sample_rate_override: Optional[int] = None,
    channels_override: Optional[int] = None,
) -> str:
    """Build JSON text frame that `audio_streamer_glue.cpp::processMessage` accepts.

    Expected schema (per your custom mod):

      {"type":"streamAudio","data":{
          "audioDataType":"raw",
          "audioData":"<base64>",
          "sampleRate":16000,
          "channels":1
      }}

    Notes:
    - keys MUST be audioDataType/audioData/sampleRate/channels.
    - mod enforces audioDataType == "raw".
    - sampleRate is required.
    """

    sample_rate = int(sample_rate_override if sample_rate_override is not None else contract.sample_rate)
    channels = int(channels_override if channels_override is not None else contract.channels)

    data = {
        "audioDataType": "raw",
        "audioData": b64encode_pcm16(frame_pcm16),
        "sampleRate": sample_rate,
        "channels": channels,
    }
    return json.dumps({"type": "streamAudio", "data": data}, separators=(",", ":"))


def fs_handshake_json(contract: FsAudioContract) -> str:
    """Optional handshake for your app/debugging (mod ignores unknown types)."""
    return json.dumps(
        {
            "type": "start",
            "sampleRate": int(contract.sample_rate),
            "channels": int(contract.channels),
            "frameMs": int(contract.frame_ms),
        },
        separators=(",", ":"),
    )
