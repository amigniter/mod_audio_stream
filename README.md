# mod_audio_stream

A FreeSWITCH module that streams L16 audio from a channel to a websocket endpoint. If websocket sends back responses (eg. JSON) it can be effectively used with ASR engines such as IBM Watson etc., or any other purpose you find applicable.

### Update (22/2/2025)

#### :rocket: **Introducing Bi-Directional Streaming with automatic playback**

A new version `mod-audio-stream v1.0.3` has been published, featuring **raw binary stream** from the websocket.
It can be downloaded from the **Releases** section (pre-release) and comes as a pre-built Debian 12 package.

- Playback feature allows continuous forward streaming while the playback runs independently.
- It is a **full-duplex streamer** between the caller and the websocket.
- It supports **base64 encoded audio** as well as the **raw binary stream** from the websocket.
- Playback can be **tracked, paused, or resumed** dynamically.

:small_blue_diamond: This release is a commercial product that is available for **free use**, including commercial use, with a limitation of **10 concurrent streaming channels**. For users requiring more than 10 channels, or access to the source code, please [contact us](mailto:amsoftswitch@gmail.com)
 for further information and licensing options.

### About

- The purpose of `mod_audio_stream` was to provide a simple, low-dependency yet effective module for streaming audio and receiving responses from a websocket server.
- Introduced [libwsc](https://github.com/amigniter/libwsc), our in-house, **RFC-6455 compliant** websocket client developed specifically for `mod_audio_stream`.
  - Replaces [ixwebsocket](https://machinezone.github.io/IXWebSocket/), which served us well for the past few years. `libwsc` is libevent-based, extremely lightweight, and optimized for low-latency audio streaming.
- This module was inspired by mod_audio_fork.

## Installation

### Dependencies
It requires `libfreeswitch-dev`, `libssl-dev`, `zlib1g-dev`, `libevent-dev` and `libspeexdsp-dev` on Debian/Ubuntu which are regular packages for Freeswitch installation.
### Building
After cloning please execute: **git submodule init** and **git submodule update** to initialize the submodule.
#### Custom path
If you built FreeSWITCH from source, eq. install dir is /usr/local/freeswitch, add path to pkgconfig:
```
export PKG_CONFIG_PATH=/usr/local/freeswitch/lib/pkgconfig
```
To build the module, from the cloned repository:
```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
sudo make install
```
**TLS** is `OFF` by default. To build with TLS support add `-DUSE_TLS=ON` to cmake line.

#### DEB Package
To build DEB package after making the module:
```
cpack -G DEB
```
Debian package will be placed in root directory `_packages` folder.

## Scripted Build & Installation

```
sudo apt-get -y install git \
    && cd /usr/src/ \
    && git clone https://github.com/amigniter/mod_audio_stream.git \
    && cd mod_audio_stream \
    && sudo bash ./build-mod-audio-stream.sh
```

### Channel variables
The following channel variables can be used to fine tune websocket connection and also configure mod_audio_stream logging:

| Variable                               | Description                                             | Default |
| -------------------------------------- | ------------------------------------------------------- | ------- |
| STREAM_MESSAGE_DEFLATE                 | true or 1, disables per message deflate                 | off     |
| STREAM_HEART_BEAT                      | number of seconds, interval to send the heart beat      | off     |
| STREAM_SUPPRESS_LOG                    | true or 1, suppresses printing to log                   | off     |
| STREAM_BUFFER_SIZE                     | buffer duration in milliseconds, divisible by 20        | 20      |
| STREAM_EXTRA_HEADERS                   | JSON object for additional headers in string format     | none    |
| ~~STREAM_NO_RECONNECT~~                    | true or 1, disables automatic websocket reconnection    | off     |
| STREAM_TLS_CA_FILE                     | CA cert or bundle, or the special values SYSTEM or NONE | SYSTEM  |
| STREAM_TLS_KEY_FILE                    | optional client key for WSS connections                 | none    |
| STREAM_TLS_CERT_FILE                   | optional client cert for WSS connections                | none    |
| STREAM_TLS_DISABLE_HOSTNAME_VALIDATION | true or 1 disable hostname check in WSS connections     | false   |

- Per message deflate compression option is enabled by default. It can lead to a very nice bandwidth savings. To disable it set the channel var to `true|1`.
- Heart beat, sent every xx seconds when there is no traffic to make sure that load balancers do not kill an idle connection.
- Suppress parameter is omitted by default(false). All the responses from websocket server will be printed to the log. Not to flood the log you can suppress it by setting the value to `true|1`. Events are fired still, it only affects printing to the log.
- `Buffer Size` actually represents a duration of audio chunk sent to websocket. If you want to send e.g. 100ms audio packets to your ws endpoint
you would set this variable to 100. If ommited, default packet size of 20ms will be sent as grabbed from the audio channel (which is default FreeSWITCH frame size)
- Extra headers should be a JSON object with key-value pairs representing additional HTTP headers. Each key should be a header name, and its corresponding value should be a string.
  ```json
  {
      "Header1": "Value1",
      "Header2": "Value2",
      "Header3": "Value3"
  }
- ~~Websocket automatic reconnection is on by default. To disable it set this channel variable to true or 1.~~
  - libwsc does not support automatic reconnection.
- TLS (for WSS) options can be fine tuned with the `STREAM_TLS_*` channel variables:
  - `STREAM_TLS_CA_FILE` the ca certificate (or certificate bundle) file. By default is `SYSTEM` which means use the system defaults.
Can be `NONE` which result in no peer verification.
  - `STREAM_TLS_CERT_FILE` optional client tls certificate file sent to the server.
  - `STREAM_TLS_KEY_FILE` optional client tls key file for the given certificate.
  - `STREAM_TLS_DISABLE_HOSTNAME_VALIDATION` if `true`, disables the check of the hostname against the peer server certificate.
Defaults to `false`, which enforces hostname match with the peer certificate.

## API

### Commands
The freeswitch module exposes the following API commands:

```
uuid_audio_stream <uuid> start <wss-url> <mix-type> <sampling-rate> <metadata>
```
Attaches a media bug and starts streaming audio (in L16 format) to the websocket server. FS default is 8k. If sampling-rate is other than 8k it will be resampled.
- `uuid` - Freeswitch channel unique id
- `wss-url` - websocket url `ws://` or `wss://`
- `mix-type` - choice of 
  - "mono" - single channel containing caller's audio
  - "mixed" - single channel containing both caller and callee audio
  - "stereo" - two channels with caller audio in one and callee audio in the other.
- `sampling-rate` - choice of
  - "8k" = 8000 Hz sample rate will be generated
  - "16k" = 16000 Hz sample rate will be generated
- `metadata` - (optional) a valid `utf-8` text to send. It will be sent the first before audio streaming starts.

```
uuid_audio_stream <uuid> send_text <metadata>
```
Sends a text to the websocket server. Requires a valid `utf-8` text.

```
uuid_audio_stream <uuid> stop <metadata>
```
Stops audio stream and closes websocket connection. If _metadata_ is provided it will be sent before the connection is closed.

```
uuid_audio_stream <uuid> pause
```
Pauses audio stream

```
uuid_audio_stream <uuid> resume
```
Resumes audio stream

## Events
Module will generate the following event types:
- `mod_audio_stream::json`
- `mod_audio_stream::connect`
- `mod_audio_stream::disconnect`
- `mod_audio_stream::error`
- `mod_audio_stream::play`

### response
Message received from websocket endpoint. Json expected, but it contains whatever the websocket server's response is.
#### Freeswitch event generated
**Name**: mod_audio_stream::json
**Body**: WebSocket server response

### connect
Successfully connected to websocket server.
#### Freeswitch event generated
**Name**: mod_audio_stream::connect
**Body**: JSON
```json
{
	"status": "connected"
}
```

### disconnect
Disconnected from websocket server.
#### Freeswitch event generated
**Name**: mod_audio_stream::disconnect
**Body**: JSON
```json
{
	"status": "disconnected",
	"message": {
		"code": 1000,
		"reason": "Normal closure"
	}
}
```
- code: `<int>`
- reason: `<string>`

### error
There is an error with the connection. Multiple fields will be available on the event to describe the error.
#### Freeswitch event generated
**Name**: mod_audio_stream::error
**Body**: JSON
```json
{
	"status": "error",
	"message": {
		"code": 1,
		"error": "String explaining the error"
	}
}
```
- code: `<int>`
- error: `<string>`

#### Possible `code` values

| Code | Enum Name             | Meaning                                              |
|:----:|:----------------------|:-----------------------------------------------------|
| 1    | `IO`                  | I/O error when reading/writing sockets               |
| 2    | `INVALID_HEADER`      | Server sent a malformed WebSocket header             |
| 3    | `SERVER_MASKED`       | Server frames were masked (not allowed by spec)      |
| 4    | `NOT_SUPPORTED`       | Requested feature (e.g. extension) not supported     |
| 5    | `PING_TIMEOUT`        | No PONG received within timeout                      |
| 6    | `CONNECT_FAILED`      | TCP connection or DNS lookup failed                  |
| 7    | `TLS_INIT_FAILED`     | Couldn't initialize SSL/TLS context                  |
| 8    | `SSL_HANDSHAKE_FAILED`| SSL/TLS handshake with server failed                 |
| 9    | `SSL_ERROR`           | Generic OpenSSL error (certificate, cipher, etc.)    |


### play
**Name**: mod_audio_stream::play
**Body**: JSON

Websocket server may return JSON object containing base64 encoded audio to be played by the user. To use this feature, response must follow the format:
```json
{
  "type": "streamAudio",
  "data": {
    "audioDataType": "raw",
    "sampleRate": 8000,
    "audioData": "base64 encoded audio"
  }
}
```
- audioDataType: `<raw|wav|mp3|ogg>`

Event generated by the module (subclass: _mod_audio_stream::play_) will be the same as the `data` element with the **file** added to it representing filePath:
```json
{
  "audioDataType": "raw",
  "sampleRate": 8000,
  "file": "/path/to/the/file"
}
```
If printing to the log is not suppressed, `response` printed to the console will look the same as the event. The original response containing base64 encoded audio is replaced because it can be quite huge.

All the files generated by this feature will reside at the temp directory and will be deleted when the session is closed.
