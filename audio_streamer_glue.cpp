#include <string>
#include <cstring>
#include "mod_audio_stream.h"
#include <ixwebsocket/IXWebSocket.h>

#include <switch_json.h>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include "base64.h"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <stdlib.h>

#define FRAME_SIZE_8000 320 /* 1000x0.02 (20ms)= 160 x(16bit= 2 bytes) 320 frame size*/
#define BUFFERIZATION_INTERVAL_MS 500

namespace
{
    extern switch_bool_t filter_json_string(switch_core_session_t *session, const char *message);
}

class AudioStreamer
{
public:
    AudioStreamer(const char *uuid, const char *wsUri, responseHandler_t callback, int deflate, int heart_beat, const char *initialMeta,
                  bool globalTrace, bool suppressLog, const char *extra_headers) : m_sessionId(uuid), m_notify(callback), m_initial_meta(initialMeta),
                                                                                   m_global_trace(globalTrace), m_suppress_log(suppressLog), m_extra_headers(extra_headers), m_playFile(0)
    {

        ix::WebSocketHttpHeaders headers;
        if (m_extra_headers)
        {
            cJSON *headers_json = cJSON_Parse(m_extra_headers);
            if (headers_json)
            {
                cJSON *iterator = headers_json->child;
                while (iterator)
                {
                    if (iterator->type == cJSON_String && iterator->valuestring != nullptr)
                    {
                        headers[iterator->string] = iterator->valuestring;
                    }
                    iterator = iterator->next;
                }
                cJSON_Delete(headers_json);
            }
        }

        webSocket.setUrl(wsUri);

        // Optional heart beat, sent every xx seconds when there is not any traffic
        // to make sure that load balancers do not kill an idle connection.
        if (heart_beat)
            webSocket.setPingInterval(heart_beat);

        // Per message deflate connection is enabled by default. You can tweak its parameters or disable it
        if (deflate)
            webSocket.disablePerMessageDeflate();

        // Set extra headers if any
        if (!headers.empty())
            webSocket.setExtraHeaders(headers);

        // Setup a callback to be fired when a message or an event (open, close, error) is received
        webSocket.setOnMessageCallback([this](const ix::WebSocketMessagePtr &msg)
                                       {
            if (msg->type == ix::WebSocketMessageType::Message)
            {
                eventCallback(MESSAGE, msg->str.c_str());

            } else if (msg->type == ix::WebSocketMessageType::Open)
            {
                cJSON *root;
                root = cJSON_CreateObject();
                cJSON_AddStringToObject(root, "status", "connected");
                char *json_str = cJSON_PrintUnformatted(root);

                eventCallback(CONNECT_SUCCESS, json_str);

                cJSON_Delete(root);
                switch_safe_free(json_str);

            } else if (msg->type == ix::WebSocketMessageType::Error)
            {
                //A message will be fired when there is an error with the connection. The message type will be ix::WebSocketMessageType::Error.
                // Multiple fields will be available on the event to describe the error.
                cJSON *root, *message;
                root = cJSON_CreateObject();
                cJSON_AddStringToObject(root, "status", "error");
                message = cJSON_CreateObject();
                cJSON_AddNumberToObject(message, "retries", msg->errorInfo.retries);
                cJSON_AddStringToObject(message, "error", msg->errorInfo.reason.c_str());
                cJSON_AddNumberToObject(message, "wait_time", msg->errorInfo.wait_time);
                cJSON_AddNumberToObject(message, "http_status", msg->errorInfo.http_status);
                cJSON_AddItemToObject(root, "message", message);

                char *json_str = cJSON_PrintUnformatted(root);

                eventCallback(CONNECT_ERROR, json_str);

                cJSON_Delete(root);
                switch_safe_free(json_str);
            }
            else if (msg->type == ix::WebSocketMessageType::Close)
            {
                // The server can send an explicit code and reason for closing.
                // This data can be accessed through the closeInfo object.
                cJSON *root, *message;
                root = cJSON_CreateObject();
                cJSON_AddStringToObject(root, "status", "disconnected");
                message = cJSON_CreateObject();
                cJSON_AddNumberToObject(message, "code", msg->closeInfo.code);
                cJSON_AddStringToObject(message, "reason", msg->closeInfo.reason.c_str());
                cJSON_AddItemToObject(root, "message", message);
                char *json_str = cJSON_PrintUnformatted(root);

                eventCallback(CONNECTION_DROPPED, json_str);

                cJSON_Delete(root);
                switch_safe_free(json_str);
            } });

        // Now that our callback is setup, we can start our background thread and receive messages
        webSocket.start();
    }

    static void media_bug_close(switch_core_session_t *session)
    {
        switch_channel_t *channel = switch_core_session_get_channel(session);
        auto *bug = (switch_media_bug_t *)switch_channel_get_private(channel, MY_BUG_NAME);
        if (bug)
            switch_core_media_bug_close(&bug, SWITCH_FALSE);
    }

    void eventCallback(notifyEvent_t event, const char *message)
    {
        switch_core_session_t *psession = switch_core_session_locate(m_sessionId.c_str());
        if (psession)
        {
            switch (event)
            {
            case CONNECT_SUCCESS:
                if (m_initial_meta && strlen(m_initial_meta) > 0)
                {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG,
                                      "sending initial metadata %s\n", m_initial_meta);
                    writeText(m_initial_meta);
                }
                m_notify(psession, EVENT_CONNECT, message);
                break;
            case CONNECTION_DROPPED:
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO, "connection closed\n");
                m_notify(psession, EVENT_DISCONNECT, message);
                break;
            case CONNECT_ERROR:
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO, "connection error\n");
                m_notify(psession, EVENT_ERROR, message);

                media_bug_close(psession);

                break;
            case MESSAGE:
                std::string msg(message);
                if (processMessage(psession, msg) != SWITCH_TRUE)
                {
                    m_notify(psession, EVENT_JSON, msg.c_str());
                }
                if (!m_suppress_log)
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG, "response: %s\n", msg.c_str());
                break;
            }
            switch_core_session_rwunlock(psession);
        }
    }

    switch_bool_t processMessage(switch_core_session_t *session, std::string &message)
    {
        cJSON *json = cJSON_Parse(message.c_str());
        switch_bool_t status = SWITCH_FALSE;
        if (!json)
        {
            return status;
        }
        const char *jsType = cJSON_GetObjectCstr(json, "type");
        if (jsType && strcmp(jsType, "streamAudio") == 0)
        {
            cJSON *jsonData = cJSON_GetObjectItem(json, "data");
            if (jsonData)
            {
                cJSON *jsonFile = nullptr;
                cJSON *jsonAudio = cJSON_DetachItemFromObject(jsonData, "audioData");
                const char *jsAudioDataType = cJSON_GetObjectCstr(jsonData, "audioDataType");
                std::string fileType;
                int sampleRate;
                if (0 == strcmp(jsAudioDataType, "raw"))
                {
                    cJSON *jsonSampleRate = cJSON_GetObjectItem(jsonData, "sampleRate");
                    sampleRate = jsonSampleRate && jsonSampleRate->valueint ? jsonSampleRate->valueint : 0;
                    std::unordered_map<int, const char *> sampleRateMap = {
                        {8000, ".r8"},
                        {16000, ".r16"},
                        {24000, ".r24"},
                        {32000, ".r32"},
                        {48000, ".r48"},
                        {64000, ".r64"}};
                    auto it = sampleRateMap.find(sampleRate);
                    fileType = (it != sampleRateMap.end()) ? it->second : "";
                }
                else if (0 == strcmp(jsAudioDataType, "wav"))
                {
                    fileType = ".wav";
                }
                else
                {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "(%s) processMessage - unsupported audio type: %s\n",
                                      m_sessionId.c_str(), jsAudioDataType);
                }

                if (jsonAudio && jsonAudio->valuestring != nullptr && !fileType.empty())
                {
                    char filePath[256];
                    std::string rawAudio = base64_decode(jsonAudio->valuestring);
                    switch_snprintf(filePath, 256, "%s%s%s_%d.tmp%s", SWITCH_GLOBAL_dirs.temp_dir,
                                    SWITCH_PATH_SEPARATOR, m_sessionId.c_str(), m_playFile++, fileType.c_str());
                    std::ofstream fstream(filePath, std::ofstream::binary);
                    fstream << rawAudio;
                    fstream.close();
                    m_Files.insert(filePath);
                    jsonFile = cJSON_CreateString(filePath);
                    cJSON_AddItemToObject(jsonData, "file", jsonFile);
                }

                if (jsonFile)
                {
                    char *jsonString = cJSON_PrintUnformatted(jsonData);
                    m_notify(session, EVENT_PLAY, jsonString);
                    message.assign(jsonString);
                    free(jsonString);
                    status = SWITCH_TRUE;
                }
                if (jsonAudio)
                    cJSON_Delete(jsonAudio);
            }
            else
            {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "(%s) processMessage - no data in streamAudio\n", m_sessionId.c_str());
            }
        }
        cJSON_Delete(json);
        return status;
    }

    ~AudioStreamer() = default;

    void disconnect()
    {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "disconnecting...\n");
        webSocket.stop();
    }

    bool isConnected()
    {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "AudioStreamer: checking connection\n");
        return (webSocket.getReadyState() == ix::ReadyState::Open);
    }

    void writeBinary(uint8_t *buffer, size_t len)
    {
        if (!this->isConnected())
            return;
        webSocket.sendBinary(ix::IXWebSocketSendData((char *)buffer, len));
    }

    void writeText(const char *text)
    {
        if (!this->isConnected())
            return;
        webSocket.sendUtf8Text(ix::IXWebSocketSendData(text, strlen(text)));
    }

    void deleteFiles()
    {
        if (m_playFile > 0)
        {
            for (const auto &fileName : m_Files)
            {
                remove(fileName.c_str());
            }
        }
    }

private:
    std::string m_sessionId;
    responseHandler_t m_notify;
    ix::WebSocket webSocket;
    const char *m_initial_meta;
    bool m_suppress_log;
    bool m_global_trace;
    const char *m_extra_headers;
    int m_playFile;
    std::unordered_set<std::string> m_Files;
};

class TcpStreamer
{
public:
    TcpStreamer(const char *uuid, const char *address, int port, responseHandler_t callback) : m_sessionId(uuid), m_address(address), m_port(port), m_notify(callback), m_socket(-1)
    {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "TcpStreamer: Initializing TCP connection to %s:%d\n", address, port);

        m_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (m_socket == -1)
        {
            std::cerr << "Could not create socket" << std::endl;
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR, "TcpStreamer: Could not create socket\n");
            return;
        }

        struct sockaddr_in server;
        server.sin_addr.s_addr = inet_addr(address);
        server.sin_family = AF_INET;
        server.sin_port = htons(port);

        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "TcpStreamer: trying to connect...\n");

        if (connect(m_socket, (struct sockaddr *)&server, sizeof(server)) < 0)
        {
            std::cerr << "Connection failed" << std::endl;
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR, "TcpStreamer: Connection to %s:%d failed\n", address, port);
            close(m_socket);
            m_socket = -1;
            return;
        }

        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "TcpStreamer: Connected to %s:%d\n", address, port);
    }

    ~TcpStreamer()
    {
        if (m_socket != -1)
        {
            close(m_socket);
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "TcpStreamer: Connection closed\n");
        }
    }

    bool isConnected()
    {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "TcpStreamer: checking connection\n");
        return m_socket != -1;
    }

    void writeBinary(uint8_t *buffer, size_t len)
    {
        if (this->isConnected())
        {
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "TcpStreamer: Sending %zu bytes\n", len);
            send(m_socket, buffer, len, 0);
        }
    }

private:
    std::string m_sessionId;
    const char *m_address;
    int m_port;
    responseHandler_t m_notify;
    int m_socket;
};

namespace
{
    bool sentAlready = false;
    std::mutex prevMsgMutex;
    std::string prevMsg;

    switch_bool_t filter_json_string(switch_core_session_t *session, const char *message)
    {
        switch_bool_t send = SWITCH_FALSE;
        cJSON *json = cJSON_Parse(message);
        if (!json)
        {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "parse - failed parsing incoming msg as JSON: %s\n", message);
            return send;
        }

        const cJSON *partial = cJSON_GetObjectItem(json, "partial");

        if (cJSON_IsString(partial))
        {
            std::string currentMsg = partial->valuestring;
            prevMsgMutex.lock();
            if (currentMsg == prevMsg)
            {
                if (!sentAlready)
                {
                    send = SWITCH_TRUE;
                    sentAlready = true;
                }
            }
            else
            {
                prevMsg = currentMsg;
                send = SWITCH_TRUE;
            }
            prevMsgMutex.unlock();
        }
        else
        {
            send = SWITCH_TRUE;
        }
        cJSON_Delete(json);
        return send;
    }

    switch_status_t stream_data_init(private_t *tech_pvt, switch_core_session_t *session, char *address, int port,
                                     uint32_t sampling, int desiredSampling, int channels, char *metadata, responseHandler_t responseHandler,
                                     int deflate, int heart_beat, bool globalTrace, bool suppressLog, int rtp_packets, const char *extra_headers)
    {
        int err; // speex

        switch_memory_pool_t *pool = switch_core_session_get_pool(session);

        memset(tech_pvt, 0, sizeof(private_t));

        strncpy(tech_pvt->sessionId, switch_core_session_get_uuid(session), MAX_SESSION_ID);
        strncpy(tech_pvt->ws_uri, address, MAX_WS_URI); // Fixed to use address instead of undefined wsUri
        tech_pvt->sampling = desiredSampling;
        tech_pvt->responseHandler = responseHandler;
        tech_pvt->rtp_packets = rtp_packets;
        tech_pvt->channels = channels;
        tech_pvt->audio_paused = 0;

        if (metadata)
            strncpy(tech_pvt->initialMetadata, metadata, MAX_METADATA_LEN);

        size_t buflen = (FRAME_SIZE_8000 * desiredSampling / 8000 * channels * BUFFERIZATION_INTERVAL_MS / 20);

        if (strcmp(STREAM_TYPE, "TCP") == 0)
        {
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "stream_data_init: initiate TCP streamer\n");
            auto *tcpStreamer = new TcpStreamer(tech_pvt->sessionId, address, port, responseHandler);
            tech_pvt->pAudioStreamer = static_cast<void *>(tcpStreamer);
        }
        else
        {
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "stream_data_init: initiate WS streamer\n");
            auto *as = new AudioStreamer(tech_pvt->sessionId, address, responseHandler, deflate, heart_beat, metadata, globalTrace, suppressLog, extra_headers);
            tech_pvt->pAudioStreamer = static_cast<void *>(as);
        }

        switch_mutex_init(&tech_pvt->mutex, SWITCH_MUTEX_NESTED, pool);

        try
        {
            size_t adjSize = 1; // adjust the buffer size to the closest pow2 size
            while (adjSize < buflen)
            {
                adjSize *= 2;
            }
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "%s: initializing buffer(%zu) to adjusted %zu bytes\n",
                              tech_pvt->sessionId, buflen, adjSize);
            tech_pvt->data = (uint8_t *)switch_core_alloc(pool, adjSize);
            tech_pvt->buffer = (RingBuffer *)switch_core_alloc(pool, sizeof(RingBuffer));
            ringBufferInit(tech_pvt->buffer, tech_pvt->data, adjSize);
        }
        catch (std::exception &e)
        {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "%s: Error initializing buffer: %s.\n",
                              tech_pvt->sessionId, e.what());
            return SWITCH_STATUS_FALSE;
        }

        if (desiredSampling != sampling)
        {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) resampling from %u to %u\n", tech_pvt->sessionId, sampling, desiredSampling);
            tech_pvt->resampler = speex_resampler_init(channels, sampling, desiredSampling, SWITCH_RESAMPLE_QUALITY, &err);
            if (0 != err)
            {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "Error initializing resampler: %s.\n", speex_resampler_strerror(err));
                return SWITCH_STATUS_FALSE;
            }
        }
        else
        {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) no resampling needed for this call\n", tech_pvt->sessionId);
        }

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) stream_data_init\n", tech_pvt->sessionId);

        return SWITCH_STATUS_SUCCESS;
    }

    void destroy_tech_pvt(private_t *tech_pvt)
    {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "%s destroy_tech_pvt\n", tech_pvt->sessionId);
        if (tech_pvt->resampler)
        {
            speex_resampler_destroy(tech_pvt->resampler);
            tech_pvt->resampler = nullptr;
        }
        if (tech_pvt->mutex)
        {
            switch_mutex_destroy(tech_pvt->mutex);
            tech_pvt->mutex = nullptr;
        }
        if (tech_pvt->pAudioStreamer)
        {
            auto *as = (AudioStreamer *)tech_pvt->pAudioStreamer;
            delete as;
            tech_pvt->pAudioStreamer = nullptr;
        }
    }

    void finish(private_t *tech_pvt)
    {
        std::shared_ptr<AudioStreamer> aStreamer;
        aStreamer.reset((AudioStreamer *)tech_pvt->pAudioStreamer);
        tech_pvt->pAudioStreamer = nullptr;

        std::thread t([aStreamer]
                      { aStreamer->disconnect(); });
        t.detach();
    }

}

extern "C"
{
    int validate_ws_uri(const char *url, char *wsUri)
    {
        const char *scheme = nullptr;
        const char *hostStart = nullptr;
        const char *hostEnd = nullptr;
        const char *portStart = nullptr;

        // Check scheme
        if (strncmp(url, "ws://", 5) == 0)
        {
            scheme = "ws";
            hostStart = url + 5;
        }
        else if (strncmp(url, "wss://", 6) == 0)
        {
            scheme = "wss";
            hostStart = url + 6;
        }
        else
        {
            return 0;
        }

        // Find host end or port start
        hostEnd = hostStart;
        while (*hostEnd && *hostEnd != ':' && *hostEnd != '/')
        {
            if (!std::isalnum(*hostEnd) && *hostEnd != '-' && *hostEnd != '.')
            {
                return 0;
            }
            ++hostEnd;
        }

        // Check if host is empty
        if (hostStart == hostEnd)
        {
            return 0;
        }

        // Check for port
        if (*hostEnd == ':')
        {
            portStart = hostEnd + 1;
            while (*portStart && *portStart != '/')
            {
                if (!std::isdigit(*portStart))
                {
                    return 0;
                }
                ++portStart;
            }
        }

        // Copy valid URI to wsUri
        std::strncpy(wsUri, url, MAX_WS_URI);
        return 1;
    }

    switch_status_t is_valid_utf8(const char *str)
    {
        switch_status_t status = SWITCH_STATUS_FALSE;
        while (*str)
        {
            if ((*str & 0x80) == 0x00)
            {
                // 1-byte character
                str++;
            }
            else if ((*str & 0xE0) == 0xC0)
            {
                // 2-byte character
                if ((str[1] & 0xC0) != 0x80)
                {
                    return status;
                }
                str += 2;
            }
            else if ((*str & 0xF0) == 0xE0)
            {
                // 3-byte character
                if ((str[1] & 0xC0) != 0x80 || (str[2] & 0xC0) != 0x80)
                {
                    return status;
                }
                str += 3;
            }
            else if ((*str & 0xF8) == 0xF0)
            {
                // 4-byte character
                if ((str[1] & 0xC0) != 0x80 || (str[2] & 0xC0) != 0x80 || (str[3] & 0xC0) != 0x80)
                {
                    return status;
                }
                str += 4;
            }
            else
            {
                // invalid character
                return status;
            }
        }
        return SWITCH_STATUS_SUCCESS;
    }

    int return_port(const char *address)
    {
        const char *hostStart = nullptr;
        const char *hostEnd = nullptr;
        const char *portStart = nullptr;
        int newPort = 0;

        hostStart = address;
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "return_port: hostStart = %s\n", hostStart);
        hostEnd = address;
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "return_port: hostEnd = %s\n", hostEnd);
        while (*hostEnd && *hostEnd != ':')
        {
            if (!std::isalnum(*hostEnd) && *hostEnd != '-' && *hostEnd != '.')
            {
                return 0;
            }
            ++hostEnd;
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "return_port: hostEnd increased and now = %s\n", hostEnd);
        }
        if (*hostEnd == ':')
        {
            portStart = hostEnd + 1;
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "return_port: portStart  = %s\n", portStart);
            while (*portStart && *portStart != '/')
            {
                if (!std::isdigit(*portStart))
                {
                    return 0;
                }
                ++portStart;
                switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "return_port: portStart increased and now = %s\n", portStart);
            }
            const char *portString = hostEnd + 1;
            newPort = atoi(portString);
        }
        return newPort;
    }

    int validate_address(const char *address, char *wsUri, char *tcpAddress, int *port) // Corrected port type
    {
        const char *scheme = nullptr;
        const char *hostStart = nullptr;
        const char *hostEnd = nullptr;
        const char *portStart = nullptr;

        // Check scheme for WS
        if (strncmp(address, "ws://", 5) == 0 || strncmp(address, "wss://", 6) == 0)
        {
            return validate_ws_uri(address, wsUri);
        }
        else
        {
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "validate_address: it is not a WS address\n");
            return 0;
        }
    }

    switch_status_t stream_session_send_text(switch_core_session_t *session, char *text)
    {
        switch_channel_t *channel = switch_core_session_get_channel(session);
        auto *bug = (switch_media_bug_t *)switch_channel_get_private(channel, MY_BUG_NAME);
        if (!bug)
        {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "stream_session_send_text failed because no bug\n");
            return SWITCH_STATUS_FALSE;
        }
        auto *tech_pvt = (private_t *)switch_core_media_bug_get_user_data(bug);

        if (!tech_pvt)
            return SWITCH_STATUS_FALSE;
        auto *pAudioStreamer = static_cast<AudioStreamer *>(tech_pvt->pAudioStreamer);
        if (pAudioStreamer && text)
            pAudioStreamer->writeText(text);

        return SWITCH_STATUS_SUCCESS;
    }

    switch_status_t stream_session_pauseresume(switch_core_session_t *session, int pause)
    {
        switch_channel_t *channel = switch_core_session_get_channel(session);
        auto *bug = (switch_media_bug_t *)switch_channel_get_private(channel, MY_BUG_NAME);
        if (!bug)
        {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "stream_session_pauseresume failed because no bug\n");
            return SWITCH_STATUS_FALSE;
        }
        auto *tech_pvt = (private_t *)switch_core_media_bug_get_user_data(bug);

        if (!tech_pvt)
            return SWITCH_STATUS_FALSE;

        switch_core_media_bug_flush(bug);
        tech_pvt->audio_paused = pause;
        return SWITCH_STATUS_SUCCESS;
    }

    switch_status_t stream_session_init(switch_core_session_t *session,
                                        responseHandler_t responseHandler,
                                        uint32_t samples_per_second,
                                        char *address,
                                        int port,
                                        int sampling,
                                        int channels,
                                        char *metadata,
                                        void **ppUserData,
                                        const char *streamType)
    {
        int deflate = 0, heart_beat = 0;
        bool globalTrace = false;
        bool suppressLog = false;
        const char *buffer_size;
        const char *extra_headers;
        int rtp_packets = 1;

        switch_channel_t *channel = switch_core_session_get_channel(session);

        if (switch_channel_var_true(channel, "STREAM_MESSAGE_DEFLATE"))
        {
            deflate = 1;
        }

        if (switch_channel_var_true(channel, "STREAM_GLOBAL_TRACE"))
        {
            globalTrace = true;
        }

        if (switch_channel_var_true(channel, "STREAM_SUPPRESS_LOG"))
        {
            suppressLog = true;
        }

        const char *heartBeat = switch_channel_get_variable(channel, "STREAM_HEART_BEAT");
        if (heartBeat)
        {
            char *endptr;
            long value = strtol(heartBeat, &endptr, 10);
            if (*endptr == '\0' && value <= INT_MAX && value >= INT_MIN)
            {
                heart_beat = (int)value;
            }
        }

        if ((buffer_size = switch_channel_get_variable(channel, "STREAM_BUFFER_SIZE")))
        {
            int bSize = atoi(buffer_size);
            if (bSize % 20 != 0)
            {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_WARNING, "%s: Buffer size of %s is not a multiple of 20ms. Using default 20ms.\n",
                                  switch_channel_get_name(channel), buffer_size);
            }
            else if (bSize >= 20)
            {
                rtp_packets = bSize / 20;
            }
        }

        extra_headers = switch_channel_get_variable(channel, "STREAM_EXTRA_HEADERS");

        // allocate per-session tech_pvt
        auto *tech_pvt = (private_t *)switch_core_session_alloc(session, sizeof(private_t));

        if (!tech_pvt)
        {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "error allocating memory!\n");
            return SWITCH_STATUS_FALSE;
        }
        if (SWITCH_STATUS_SUCCESS != stream_data_init(tech_pvt, session, address, port, samples_per_second, sampling, channels, metadata, responseHandler, deflate, heart_beat,
                                                      globalTrace, suppressLog, rtp_packets, extra_headers))
        {
            destroy_tech_pvt(tech_pvt);
            return SWITCH_STATUS_FALSE;
        }

        *ppUserData = tech_pvt;

        return SWITCH_STATUS_SUCCESS;
    }

    switch_bool_t stream_frame(switch_media_bug_t *bug)
    {
        // switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "stream_frame: trying to stream\n");
        auto *tech_pvt = (private_t *)switch_core_media_bug_get_user_data(bug);
        if (!tech_pvt || tech_pvt->audio_paused)
            return SWITCH_TRUE;

        if (switch_mutex_trylock(tech_pvt->mutex) == SWITCH_STATUS_SUCCESS)
        {
            if (!tech_pvt->pAudioStreamer)
            {
                switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "stream_frame: no audio streamer\n");
                switch_mutex_unlock(tech_pvt->mutex);
                return SWITCH_TRUE;
            }

            auto *pAudioStreamer = static_cast<AudioStreamer *>(tech_pvt->pAudioStreamer);
            auto *pTcpStreamer = static_cast<TcpStreamer *>(tech_pvt->pAudioStreamer);

            if (strcmp(STREAM_TYPE, "TCP") == 0 ? !pTcpStreamer->isConnected() : !pAudioStreamer->isConnected())
            {
                switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "stream_frame: audio streamer is not connected\n");
                switch_mutex_unlock(tech_pvt->mutex);
                return SWITCH_TRUE;
            }

            uint8_t data[SWITCH_RECOMMENDED_BUFFER_SIZE];
            switch_frame_t frame = {};
            frame.data = data;
            frame.buflen = SWITCH_RECOMMENDED_BUFFER_SIZE;
            size_t available = ringBufferFreeSpace(tech_pvt->buffer);

            while (switch_core_media_bug_read(bug, &frame, SWITCH_TRUE) == SWITCH_STATUS_SUCCESS)
            {
                if (frame.datalen)
                {
                    size_t remaining = 0;
                    if (available >= frame.datalen)
                    {
                        ringBufferAppendMultiple(tech_pvt->buffer, static_cast<uint8_t *>(frame.data), frame.datalen);
                    }
                    else
                    {
                        ringBufferAppendMultiple(tech_pvt->buffer, static_cast<uint8_t *>(frame.data), available);
                        remaining = frame.datalen - available;
                    }

                    if (ringBufferLen(tech_pvt->buffer) >= FRAME_SIZE_8000 * tech_pvt->sampling / 8000 * tech_pvt->channels * BUFFERIZATION_INTERVAL_MS / 20)
                    {
                        size_t nFrames = ringBufferLen(tech_pvt->buffer);
                        uint8_t chunkPtr[nFrames];
                        ringBufferGetMultiple(tech_pvt->buffer, chunkPtr, nFrames);

                        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "stream_frame: writing binary\n");
                        strcmp(STREAM_TYPE, "TCP") == 0 ? pTcpStreamer->writeBinary(chunkPtr, nFrames) : pAudioStreamer->writeBinary(chunkPtr, nFrames);
                        ringBufferClear(tech_pvt->buffer);
                    }
                }
            }
            switch_mutex_unlock(tech_pvt->mutex);
        }
        return SWITCH_TRUE;
    }

    switch_status_t stream_session_cleanup(switch_core_session_t *session, char *text, int channelIsClosing)
    {
        switch_channel_t *channel = switch_core_session_get_channel(session);
        auto *bug = (switch_media_bug_t *)switch_channel_get_private(channel, MY_BUG_NAME);
        if (bug)
        {
            auto *tech_pvt = (private_t *)switch_core_media_bug_get_user_data(bug);
            char sessionId[MAX_SESSION_ID];
            strcpy(sessionId, tech_pvt->sessionId);

            switch_mutex_lock(tech_pvt->mutex);
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) stream_session_cleanup\n", sessionId);

            switch_channel_set_private(channel, MY_BUG_NAME, nullptr);
            if (!channelIsClosing)
            {
                switch_core_media_bug_remove(session, &bug);
            }

            auto *audioStreamer = (AudioStreamer *)tech_pvt->pAudioStreamer;
            if (audioStreamer)
            {
                audioStreamer->deleteFiles();
                if (text)
                    audioStreamer->writeText(text);
                finish(tech_pvt);
            }

            destroy_tech_pvt(tech_pvt);

            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "(%s) stream_session_cleanup: connection closed\n", sessionId);
            return SWITCH_STATUS_SUCCESS;
        }

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "stream_session_cleanup: no bug - websocket connection already closed\n");
        return SWITCH_STATUS_FALSE;
    }
}
