#ifndef AUDIO_STREAMER_GLUE_H
#define AUDIO_STREAMER_GLUE_H
#include "mod_audio_stream.h"

int validate_ws_uri(const char *url, char *wsUri);
int validate_address(const char *address, char *wsUri, char *tcpAddress);
switch_status_t is_valid_utf8(const char *str);
switch_status_t stream_session_send_text(switch_core_session_t *session, char *text);
switch_status_t stream_session_pauseresume(switch_core_session_t *session, int pause);
switch_status_t stream_session_init(switch_core_session_t *session, responseHandler_t responseHandler,
                                    uint32_t samples_per_second, char *wsUri, int port, int sampling, int channels, char *metadata, void **ppUserData);
switch_bool_t stream_frame(switch_media_bug_t *bug);
switch_status_t stream_session_cleanup(switch_core_session_t *session, char *text, int channelIsClosing);

class TcpStreamer
{
public:
    TcpStreamer(const char *uuid, const char *address, int port, responseHandler_t callback);
    ~TcpStreamer();
    bool isConnected();
    void writeBinary(uint8_t *buffer, size_t len);

private:
    std::string m_sessionId;
    const char *m_address;
    int m_port;
    responseHandler_t m_notify;
    int m_socket;
};

#endif // AUDIO_STREAMER_GLUE_H
