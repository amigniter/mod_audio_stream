#include <string>
#include <cstring>
#include "mod_audio_stream.h"
#include "streamer.h"

#define FRAME_SIZE_8000  320 /* 1000x0.02 (20ms)= 160 x(16bit= 2 bytes) 320 frame size*/

namespace {

    switch_status_t stream_data_init(private_t *tech_pvt, switch_core_session_t *session, char *wsUri,
                                     uint32_t sampling, int desiredSampling, int channels, responseHandler_t responseHandler,
                                     shared_wrapper *sharedWrapper, bool suppressLog, int rtp_packets)
    {
        int err; //speex

        switch_memory_pool_t *pool = switch_core_session_get_pool(session);

        memset(tech_pvt, 0, sizeof(private_t));

        strncpy(tech_pvt->sessionId, switch_core_session_get_uuid(session), MAX_SESSION_ID);
        strncpy(tech_pvt->ws_uri, wsUri, MAX_WS_URI);
        tech_pvt->sampling = desiredSampling;
        tech_pvt->responseHandler = responseHandler;
        tech_pvt->rtp_packets = rtp_packets;
        tech_pvt->channels = channels;
        tech_pvt->audio_paused = 0;

        size_t buflen = (FRAME_SIZE_8000 * desiredSampling / 8000 * channels * rtp_packets);

        std::shared_ptr<AudioStreamer> aStreamer = std::make_shared<AudioStreamer>(tech_pvt->sessionId, wsUri, responseHandler, suppressLog);
        auto *pSharedWrapper = new (sharedWrapper) shared_wrapper(aStreamer);

        tech_pvt->pSharedStreamer = static_cast<void *> (pSharedWrapper);

        switch_mutex_init(&tech_pvt->mutex, SWITCH_MUTEX_NESTED, pool);

        try {
            size_t adjSize = 1; //adjust the buffer size to the closest pow2 size
            while(adjSize < buflen) {
                adjSize *=2;
            }
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "%s: initializing buffer(%zu) to adjusted %zu bytes\n",
                              tech_pvt->sessionId, buflen, adjSize);
            tech_pvt->data = (uint8_t *) switch_core_alloc(pool, adjSize);
            tech_pvt->buffer = (RingBuffer *) switch_core_alloc(pool, sizeof(RingBuffer));
            ringBufferInit(tech_pvt->buffer, tech_pvt->data, adjSize);
        } catch (std::exception& e) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "%s: Error initializing buffer: %s.\n",
                              tech_pvt->sessionId, e.what());
            return SWITCH_STATUS_FALSE;
        }

        if (desiredSampling != sampling) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) resampling from %u to %u\n", tech_pvt->sessionId, sampling, desiredSampling);
            tech_pvt->resampler = speex_resampler_init(channels, sampling, desiredSampling, SWITCH_RESAMPLE_QUALITY, &err);
            if (0 != err) {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "Error initializing resampler: %s.\n", speex_resampler_strerror(err));
                return SWITCH_STATUS_FALSE;
            }
        }
        else {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) no resampling needed for this call\n", tech_pvt->sessionId);
        }

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) stream_data_init\n", tech_pvt->sessionId);

        //stream_connect(tech_pvt);
        return SWITCH_STATUS_SUCCESS;
    }

    void destroy_tech_pvt(private_t* tech_pvt) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "%s destroy_tech_pvt\n", tech_pvt->sessionId);
        if (tech_pvt->pSharedStreamer) {
            auto* sharedWrapper = (shared_wrapper *) tech_pvt->pSharedStreamer;
            sharedWrapper->~shared_wrapper();
            tech_pvt->pSharedStreamer=nullptr;
        }
        if (tech_pvt->resampler) {
            speex_resampler_destroy(tech_pvt->resampler);
            tech_pvt->resampler = nullptr;
        }
        if (tech_pvt->mutex) {
            switch_mutex_destroy(tech_pvt->mutex);
            tech_pvt->mutex = nullptr;
        }
    }

}

extern "C" {
    int validate_ws_uri(const char* url, char* wsUri) {
        const char* scheme = nullptr;
        const char* hostStart = nullptr;
        const char* hostEnd = nullptr;
        const char* portStart = nullptr;

        // Check scheme
        if (strncmp(url, "ws://", 5) == 0) {
            scheme = "ws";
            hostStart = url + 5;
        } else if (strncmp(url, "wss://", 6) == 0) {
            scheme = "wss";
            hostStart = url + 6;
        } else {
            return 0;
        }

        // Find host end or port start
        hostEnd = hostStart;
        while (*hostEnd && *hostEnd != ':' && *hostEnd != '/') {
            if (!std::isalnum(*hostEnd) && *hostEnd != '-' && *hostEnd != '.') {
                return 0;
            }
            ++hostEnd;
        }

        // Check if host is empty
        if (hostStart == hostEnd) {
            return 0;
        }

        // Check for port
        if (*hostEnd == ':') {
            portStart = hostEnd + 1;
            while (*portStart && *portStart != '/') {
                if (!std::isdigit(*portStart)) {
                    return 0;
                }
                ++portStart;
            }
        }

        // Copy valid URI to wsUri
        std::strncpy(wsUri, url, MAX_WS_URI);
        return 1;
    }

    switch_status_t stream_session_pauseresume(switch_core_session_t *session, int pause) {
        switch_channel_t *channel = switch_core_session_get_channel(session);
        auto *bug = (switch_media_bug_t*) switch_channel_get_private(channel, MY_BUG_NAME);
        if (!bug) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "stream_session_pauseresume failed because no bug\n");
            return SWITCH_STATUS_FALSE;
        }
        auto *tech_pvt = (private_t*) switch_core_media_bug_get_user_data(bug);

        if (!tech_pvt) return SWITCH_STATUS_FALSE;

        switch_core_media_bug_flush(bug);
        tech_pvt->audio_paused = pause;
        return SWITCH_STATUS_SUCCESS;
    }

    switch_status_t stream_session_init(switch_core_session_t *session,
                                        responseHandler_t responseHandler,
                                        uint32_t samples_per_second,
                                        char *wsUri,
                                        int sampling,
                                        int channels,
                                        void **ppUserData)
    {
        bool suppressLog = false;
        const char* buffer_size;
        int rtp_packets = 1;

        switch_channel_t *channel = switch_core_session_get_channel(session);

        if (switch_channel_var_true(channel, "STREAM_SUPPRESS_LOG")) {
            suppressLog = true;
        }
        if ((buffer_size = switch_channel_get_variable(channel, "STREAM_BUFFER_SIZE"))) {
            int bSize = atoi(buffer_size);
            if(bSize % 20 != 0) {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_WARNING, "%s: Buffer size of %s is not a multiple of 20ms. Using default 20ms.\n",
                                  switch_channel_get_name(channel), buffer_size);
            } else if(bSize >= 20){
                rtp_packets = bSize/20;
            }
        }
        // allocate per-session tech_pvt
        auto* tech_pvt = (private_t *) switch_core_session_alloc(session, sizeof(private_t));
        // allocate mem for shared wrapper
        auto* sharedWrapper = (shared_wrapper *) switch_core_session_alloc(session, sizeof(shared_wrapper));

        if (!tech_pvt) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "error allocating memory!\n");
            return SWITCH_STATUS_FALSE;
        }
        if (SWITCH_STATUS_SUCCESS != stream_data_init(tech_pvt, session, wsUri, samples_per_second, sampling, channels, responseHandler, sharedWrapper, suppressLog, rtp_packets)) {
            destroy_tech_pvt(tech_pvt);
            return SWITCH_STATUS_FALSE;
        }

        *ppUserData = tech_pvt;

        return SWITCH_STATUS_SUCCESS;
    }

    switch_bool_t stream_frame(switch_media_bug_t *bug)
    {
        auto* tech_pvt = (private_t*) switch_core_media_bug_get_user_data(bug);
        if (!tech_pvt || tech_pvt->audio_paused) return SWITCH_TRUE;

        switch_status_t acq_lock = switch_mutex_trylock(tech_pvt->mutex);

        if (acq_lock == SWITCH_STATUS_SUCCESS) {

            if (!tech_pvt->pSharedStreamer) {
                switch_mutex_unlock(tech_pvt->mutex);
                return SWITCH_TRUE;
            }

            auto* sharedWrapper = static_cast<shared_wrapper*>(tech_pvt->pSharedStreamer);
            std::shared_ptr<AudioStreamer> pAudioStreamer = sharedWrapper->aStreamer;

            if(!pAudioStreamer->isConnected()) {
                switch_mutex_unlock(tech_pvt->mutex);
                return SWITCH_TRUE;
            }

            if (nullptr == tech_pvt->resampler) {
                uint8_t data[SWITCH_RECOMMENDED_BUFFER_SIZE];
                switch_frame_t frame = {};
                frame.data = data;
                frame.buflen = SWITCH_RECOMMENDED_BUFFER_SIZE;
                size_t available = ringBufferFreeSpace(tech_pvt->buffer);
                while (switch_core_media_bug_read(bug, &frame, SWITCH_TRUE) == SWITCH_STATUS_SUCCESS) {
                    if(frame.datalen) {
                        if (1 == tech_pvt->rtp_packets) {
                            std::vector<uint8_t> dataVector(
                                    reinterpret_cast<uint8_t*>(frame.data),
                                    reinterpret_cast<uint8_t*>(frame.data) + frame.datalen
                            );

                            pAudioStreamer->writeBinary(dataVector);
                            continue;
                        }

                        size_t remaining = 0;
                        if(available >= frame.datalen) {
                            ringBufferAppendMultiple(tech_pvt->buffer, static_cast<uint8_t *>(frame.data), frame.datalen);
                        } else {
                            // The remaining space is not sufficient for the entire chunk
                            // so write first part up to the available space
                            ringBufferAppendMultiple(tech_pvt->buffer, static_cast<uint8_t *>(frame.data), available);
                            remaining = frame.datalen - available;
                        }

                        if(0 == ringBufferFreeSpace(tech_pvt->buffer)) {
                            size_t nFrames = ringBufferLen(tech_pvt->buffer);
                            size_t nBytes = nFrames + remaining;
                            std::vector<uint8_t> chunkVector(nBytes);
                            ringBufferGetMultiple(tech_pvt->buffer, chunkVector.data(), nBytes);

                            if(remaining > 0) {
                                size_t startIndex = frame.datalen - remaining;
                                memcpy(chunkVector.data() + nBytes - remaining, static_cast<uint8_t *>(frame.data) + startIndex, remaining);
                            }

                            pAudioStreamer->writeBinary(chunkVector);

                            ringBufferClear(tech_pvt->buffer);
                        }

                    }
                }
            } else {
                uint8_t data[SWITCH_RECOMMENDED_BUFFER_SIZE];
                switch_frame_t frame = {};
                frame.data = data;
                frame.buflen = SWITCH_RECOMMENDED_BUFFER_SIZE;
                size_t available = ringBufferFreeSpace(tech_pvt->buffer);

                while (switch_core_media_bug_read(bug, &frame, SWITCH_TRUE) == SWITCH_STATUS_SUCCESS) {
                    if(frame.datalen) {
                        spx_uint32_t in_len = frame.samples;
                        spx_uint32_t out_len = available >> 1;
                        spx_int16_t out[available];

                        speex_resampler_process_interleaved_int(tech_pvt->resampler,
                                (const spx_int16_t *)frame.data,
                                (spx_uint32_t *) &in_len,
                                &out[0],
                                &out_len);

                        size_t remaining = 0;

                        if(out_len>0) {
                            size_t bytes_written = out_len << tech_pvt->channels;
                            if (1 == tech_pvt->rtp_packets) {
                                std::vector<uint8_t> dataVector(
                                        reinterpret_cast<uint8_t*>(out),
                                        reinterpret_cast<uint8_t*>(out) + bytes_written
                                );
                                pAudioStreamer->writeBinary(dataVector);
                                continue;
                            }
                            if (bytes_written <= available) {
                                // Case 1: Resampled data fits entirely in the buffer
                                ringBufferAppendMultiple(tech_pvt->buffer, (const uint8_t *)out, bytes_written);
                            } else {
                                // Case 2: Resampled data partially fits in the buffer
                                ringBufferAppendMultiple(tech_pvt->buffer, (const uint8_t *)out, available);
                                remaining = bytes_written - available;
                            }

                            available -= bytes_written;
                            if(available <= 2) {
                                spx_uint32_t in_len_rem = frame.samples-in_len;
                                spx_uint32_t out_len_rem = SWITCH_RECOMMENDED_BUFFER_SIZE;
                                spx_int16_t out_rem[SWITCH_RECOMMENDED_BUFFER_SIZE];
                                speex_resampler_process_interleaved_int(tech_pvt->resampler,
                                                                        (const spx_int16_t *)frame.data+in_len, //in_len_rem
                                                                        (spx_uint32_t *) &in_len_rem,
                                                                        &out_rem[0],
                                                                        &out_len_rem);
                                size_t rem_bytes = out_len_rem << tech_pvt->channels;
                                //size_t rem_bytes = out_len_rem * tech_pvt->channels * sizeof(spx_int16_t);
                                size_t bufferLen = ringBufferLen(tech_pvt->buffer);
                                size_t nFrames = bufferLen + rem_bytes;
                                std::vector<uint8_t> bufferVector(nFrames);
                                ringBufferGetMultiple(tech_pvt->buffer, bufferVector.data(), bufferLen);
                                memcpy(bufferVector.data() + bufferLen, (uint8_t*)&out_rem[0], rem_bytes);
                                ringBufferClear(tech_pvt->buffer);
                                pAudioStreamer->writeBinary(bufferVector);
                            }
                        }

                    }
                }
            }
            switch_mutex_unlock(tech_pvt->mutex);
        }
        return SWITCH_TRUE;
    }

    switch_status_t stream_session_cleanup(switch_core_session_t *session, int channelIsClosing) {
        switch_channel_t *channel = switch_core_session_get_channel(session);
        auto *bug = (switch_media_bug_t*) switch_channel_get_private(channel, MY_BUG_NAME);
        if(bug)
        {
            auto* tech_pvt = (private_t*) switch_core_media_bug_get_user_data(bug);
            char sessionId[MAX_SESSION_ID];
            strcpy(sessionId, tech_pvt->sessionId);

            switch_mutex_lock(tech_pvt->mutex);
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) stream_session_cleanup\n", sessionId);

            switch_channel_set_private(channel, MY_BUG_NAME, nullptr);
            if (!channelIsClosing) {
                switch_core_media_bug_remove(session, &bug);
                switch_channel_set_private(channel, MY_BUG_NAME, nullptr);
            }

            auto* sharedWrapper = (shared_wrapper *) tech_pvt->pSharedStreamer;
            std::shared_ptr<AudioStreamer> pAudioStreamer = sharedWrapper->aStreamer;

            if(pAudioStreamer) {
                pAudioStreamer->disconnect();
            }

            destroy_tech_pvt(tech_pvt);

            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "(%s) stream_session_cleanup: connection closed\n", sessionId);
            return SWITCH_STATUS_SUCCESS;
        }

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "stream_session_cleanup: no bug - websocket connection already closed\n");
        return SWITCH_STATUS_FALSE;
    }

    switch_status_t stream_connect(void *arg) {
        auto *tech_pvt= (private_t*) arg;
        if(tech_pvt) {
            auto* sharedWrapper = static_cast<shared_wrapper*>(tech_pvt->pSharedStreamer);
            std::shared_ptr<AudioStreamer> aStreamer = sharedWrapper->aStreamer;
            aStreamer->connect(sharedWrapper);
            return SWITCH_STATUS_SUCCESS;
        }
        return SWITCH_STATUS_FALSE;
    }
}

