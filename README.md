# mod_audio_stream

A FreeSWITCH module that streams L16 audio from a channel to a websocket endpoint. If websocket sends back responses (eg. JSON) it can be effectively used with ASR engines such as IBM Watson etc., or any other purpose you find applicable.

#### About

- The purpose of `mod_audio_stream` was to make a simple, less dependent but yet effective module to stream audio and receive responses from websocket server. It uses [ixwebsocket](https://machinezone.github.io/IXWebSocket/), c++ library for websocket protocol which is compiled as a static library.
- This module was inspired by [mod_audio_fork](https://github.com/drachtio/drachtio-freeswitch-modules/tree/main/modules/mod_audio_fork).

## Installation

### Dependencies
It requires `libfreeswitch-dev`, `libssl-dev`, `zlib1g-dev` and `libspeexdsp-dev` on Debian/Ubuntu which are regular packages for Freeswitch installation.
### Building
After cloning please execute: **git submodule init** and **git submodule update** to initialize the submodule.
#### Custom path
If you built FreeSWITCH from source, eq. install dir is /usr/local/freeswitch, add path to pkgconfig:
```
export PKG_CONFIG_PATH=/usr/local/freeswitch/lib/pkgconfig
```
To build the module, from the cloned repository directory:
```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
sudo make install
```

### Channel variables
The following channel variables can be used to fine tune websocket connection and also configure mod_audio_stream logging:

| Variable               | Description                                         | Default |
|------------------------|-----------------------------------------------------|---------|
| STREAM_MESSAGE_DEFLATE | true or 1, disables per message deflate             | off     |
| STREAM_HEART_BEAT      | number of seconds, interval to send the heart beat  | off     |
| STREAM_SUPPRESS_LOG    | true or 1, suppresses printing to log               | off     |
| STREAM_BUFFER_SIZE     | buffer duration in milliseconds, divisible by 20    | 20      |
| STREAM_EXTRA_HEADERS   | JSON object for additional headers in string format | none    |

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
		"retries": 1,
		"error": "Expecting status 101 (Switching Protocol), got 403 status connecting to wss://localhost, HTTP Status line: HTTP/1.1 403 Forbidden\r\n",
		"wait_time": 100,
		"http_status": 403
	}
}
```
- retries: `<int>`
- error: `<string>`
- wait_time: `<int>`
- http_status: `<int>`
