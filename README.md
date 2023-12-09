# mod audio stream <sup>_Light_</sup>

A FreeSWITCH module that streams L16 audio from a channel to a websocket endpoint. If websocket sends back responses (eg. JSON) it can be effectively used with ASR engines such as IBM Watson etc., or any other purpose you find applicable.

**Dedicated to all FreeSWITCH enthusiasts!**

### Note
- This branch is not an upgrade to the main repository; it's a whole new version designed for users seeking minimalism and high-performance audio streaming to websockets.
- If the main repo is working fine for you there is no need to switch to `v2` unless you want to stream audio only, and need a really lightweight module for that. 

### About

- The purpose of `mod_audio_stream` was to make a simple, less dependent but yet effective module to stream audio and receive responses from websocket server. It uses [easywsclient](https://github.com/dhbaird/easywsclient), a very light c++ library for websocket protocol which is reworked to support TLS (**Mbed TLS**).
- It doesn't support sending text. This version is crafted for simplicity and efficiency, focusing only on the essentials of audio streaming.

## Installation

### Dependencies
It requires `libfreeswitch-dev`, `libmbedtls-dev` and `libspeexdsp-dev`. On Debian and Ubuntu, you can install these dependencies using the following commands:

```bash
sudo apt-get install libfreeswitch-dev libmbedtls-dev libspeexdsp-dev
```
If you are using a different Linux distribution, you'll need to install the equivalent packages. The package names may vary depending on your distribution's package manager.

### Building

#### Custom Path
If you've built the FreeSWITCH from source, eq. install dir is /usr/local/freeswitch, add path to pkgconfig:
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

| Variable | Description | Default |
| --- | ----------- |  ---|
| STREAM_SUPPRESS_LOG | true or 1, suppresses printing to log | off |
| STREAM_BUFFER_SIZE | buffer duration in milliseconds, divisible by 20 | 20 |

- Suppress parameter is omitted by default(false). All the responses from websocket server will be printed to the log. Not to flood the log you can suppress it by setting the value to `true|1`. Events are fired still, it only affects printing to the log.
- Buffer Size actually represents a duration of audio chunk sent to websocket. If you want to send e.g. 100ms audio packets to your ws endpoint you would set this variable to 100. If ommited, default packet size of 20ms will be sent as grabbed from the audio channel (which is default FreeSWITCH frame size)
## API

### Commands
The freeswitch module exposes the following API commands:

```
uuid_audio_stream <uuid> start <wss-url> <mix-type> <sampling-rate>
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

```
uuid_audio_stream <uuid> stop
```
Stops audio stream and closes websocket connection.

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
- `mod_audio_stream::disconnect`
- `mod_audio_stream::error`

### response
Message received from websocket endpoint. Json expected, but it contains whatever the websocket server's response is.
#### Freeswitch event generated
**Name**: mod_audio_stream::json
**Body**: WebSocket server response

### disconnect
Disconnected from websocket server.
#### Freeswitch event generated
**Name**: mod_audio_stream::disconnect
**Body**: JSON
```json
{
	"status": "disconnected"
}
```

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
		"error": "error description"
	}
}
```
- code: `<int>`
- error: `<string>`
