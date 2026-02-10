from bridge.fs_payloads import FsAudioContract, fs_stream_audio_json


def test_stream_audio_schema_keys():
    contract = FsAudioContract(sample_rate=16000, channels=1, frame_ms=20)
    s = fs_stream_audio_json(b"\x00\x01" * 160, contract)
    assert '"type":"streamAudio"' in s
    assert '"audioDataType":"raw"' in s
    assert '"audioData"' in s
    assert '"sampleRate":16000' in s
    assert '"channels":1' in s


def test_stream_audio_sample_rate_override():
    """Production path: contract is 8kHz but we override to 24kHz (OpenAI native rate)."""
    contract = FsAudioContract(sample_rate=8000, channels=1, frame_ms=20)
    s = fs_stream_audio_json(b"\x00\x01" * 480, contract, sample_rate_override=24000)
    assert '"sampleRate":24000' in s
    assert '"channels":1' in s
    assert '"audioDataType":"raw"' in s
