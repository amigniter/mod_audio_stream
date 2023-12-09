#ifndef STREAMER_H
#define STREAMER_H

#include "mod_audio_stream.h"
#include "easywsclient.hpp"
#include "tinythread.h"
#include <memory>
#include "switch_cJSON.h"

struct shared_wrapper;

class AudioStreamer {
public:
	AudioStreamer(const char* uuid, const char* wsUri, responseHandler_t callback, bool suppressLog);
	~AudioStreamer();

	bool connect(shared_wrapper* ptr);
    void disconnect();

    static void media_bug_close(switch_core_session_t *session);
    void eventCallback(const char* sessionId, notifyEvent_t event, const char* message) const;

	bool isConnected();

    void writeBinary(std::vector<uint8_t>&buffer);

    tthread::mutex wss_mutex;
    tthread::thread* thrd;

private:
    void ws_thread();
    static void StaticWsThread(void* arg);
    static void OnErrorCallbackWrapper(const easywsclient::WSError& err, void* userData);
    static void OnWebSocketCallbackWrapper(const std::string& message, void* userData);
    void OnWebSocketCallback(const std::string& message, void* pUserData);
    void OnErrorCallback(const easywsclient::WSError& err, void* pUserData);

	std::string m_sessionId;
	std::string m_wsUri;
    responseHandler_t m_notify;
    easywsclient::WebSocket::pointer ws;
    bool m_suppressLog;
};

struct shared_wrapper {
    std::shared_ptr<AudioStreamer> aStreamer;
    explicit shared_wrapper(std::shared_ptr<AudioStreamer>& a) : aStreamer(a) {}
};

#endif // STREAMER_H
