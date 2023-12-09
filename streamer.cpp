#include "streamer.h"

using easywsclient::WebSocket;
using tthread::lock_guard;
using tthread::mutex;

AudioStreamer::AudioStreamer(const char* uuid, const char* wsUri, responseHandler_t callback, bool suppressLog): m_sessionId(uuid), m_notify(callback), m_wsUri(wsUri),
                            m_suppressLog(suppressLog) {
    thrd = nullptr;
    ws = nullptr;
}

AudioStreamer::~AudioStreamer() {
    delete ws;
    if(thrd && thrd->joinable())
        thrd->join();
    delete thrd;
}

void AudioStreamer::OnErrorCallback(const easywsclient::WSError& err, void* pUserData) {
    if(err.code == easywsclient::WSError::CONNECTION_CLOSED) {
        cJSON *root;
        root = cJSON_CreateObject();
        cJSON_AddStringToObject(root, "status", "disconnected");
        char *json_str = cJSON_PrintUnformatted(root);
        eventCallback(m_sessionId.c_str(), DISCONNECTED, err.message.c_str());
        cJSON_Delete(root);
        switch_safe_free(json_str);
    } else {
        cJSON *root, *message;
        root = cJSON_CreateObject();
        cJSON_AddStringToObject(root, "status", "error");
        message = cJSON_CreateObject();
        cJSON_AddNumberToObject(message, "code", err.code);
        cJSON_AddStringToObject(message, "error", err.message.c_str());
        cJSON_AddItemToObject(root, "message", message);
        char *json_str = cJSON_PrintUnformatted(root);
        eventCallback(m_sessionId.c_str(), ERROR, err.message.c_str());
        cJSON_Delete(root);
        switch_safe_free(json_str);
    }
}

void AudioStreamer::OnWebSocketCallback(const std::string& message, void* pUserData)
{
    eventCallback(m_sessionId.c_str(), MESSAGE, message.c_str());
}

void AudioStreamer::eventCallback(const char* sessionId, notifyEvent_t event, const char* message) const {
    switch_core_session_t* session = switch_core_session_locate(sessionId);
    if(session){
        switch_channel_t *channel = switch_core_session_get_channel(session);
        if(channel) {
            switch (event) {
                case MESSAGE:
                    if(!m_suppressLog)
                        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "response: %s\n", message);
                    m_notify(session, EVENT_JSON, message);
                    break;
                case DISCONNECTED:
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "connection closed\n");
                    m_notify(session, EVENT_DISCONNECT, message);
                    break;
                case ERROR:
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "connection error (%s)\n", message);
                    m_notify(session, EVENT_ERROR, message);
                    media_bug_close(session);
                    break;
            }
        }
        switch_core_session_rwunlock(session);
    }
}

void AudioStreamer::ws_thread () {
    while(ws->getReadyState() != WebSocket::CLOSED) {
        ws->poll(0, OnErrorCallbackWrapper, this);
        ws->dispatch(OnWebSocketCallbackWrapper, OnErrorCallbackWrapper, this);
    }
    disconnect();
}

// static member function to serve as an intermediary
void AudioStreamer::StaticWsThread(void* arg) {
    auto* sharedData = static_cast<shared_wrapper*>(arg);
    std::shared_ptr<AudioStreamer> aStreamer = sharedData->aStreamer;
    aStreamer->ws_thread();
}

void AudioStreamer::OnErrorCallbackWrapper(const easywsclient::WSError& err, void* userData){
    auto* instance = static_cast<AudioStreamer*>(userData);
    instance->OnErrorCallback(err, userData);
}

void AudioStreamer::OnWebSocketCallbackWrapper(const std::string& message, void* userData) {
    auto* instance = static_cast<AudioStreamer*>(userData);
    instance->OnWebSocketCallback(message, userData);
}

bool AudioStreamer::connect(shared_wrapper* ptr) {
    lock_guard<mutex> guard(wss_mutex);
    ws = WebSocket::from_url(m_wsUri, "");
    if(ws) {
        ws->connect();
        thrd = new tthread::thread(&AudioStreamer::StaticWsThread, ptr);
        return true;
    }
    return false;
}

void AudioStreamer::disconnect() {
    lock_guard<mutex> guard(wss_mutex);
    if(ws->getReadyState() != WebSocket::CLOSED)
        ws->close();
}

bool AudioStreamer::isConnected() {
    return (ws->getReadyState() == WebSocket::OPEN);
}

void AudioStreamer::writeBinary(std::vector<uint8_t> &buffer) {
    if(!this->isConnected()) return;
    lock_guard<mutex> guard(wss_mutex);
    ws->sendBinary(buffer);
}

void AudioStreamer::media_bug_close(switch_core_session_t *session) {
    switch_channel_t *channel = switch_core_session_get_channel(session);
    auto *bug = (switch_media_bug_t *) switch_channel_get_private(channel, MY_BUG_NAME);
    if(bug) switch_core_media_bug_close(&bug, SWITCH_FALSE);
}
