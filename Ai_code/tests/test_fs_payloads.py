from bridge.fs_payloads import FsAudioContract, fs_stream_audio_json


def test_stream_audio_schema_keys():
    contract = FsAudioContract(sample_rate=16000, channels=1, frame_ms=20)
    s = fs_stream_audio_json(b"\x00\x01" * 160, contract)
    assert '"type":"streamAudio"' in s
    assert '"audioDataType":"raw"' in s
    assert '"audioData"' in s
    assert '"sampleRate":16000' in s
    assert '"channels":1' in s
