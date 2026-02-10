#include "mod_audio_stream.h"
#include "audio_streamer_glue.h"

SWITCH_MODULE_SHUTDOWN_FUNCTION(mod_audio_stream_shutdown);
SWITCH_MODULE_LOAD_FUNCTION(mod_audio_stream_load);

SWITCH_MODULE_DEFINITION(
    mod_audio_stream,
    mod_audio_stream_load,
    mod_audio_stream_shutdown,
    NULL
);

static void responseHandler(switch_core_session_t* session,
                            const char* eventName,
                            const char* json)
{
    switch_event_t *event;
    switch_channel_t *channel = switch_core_session_get_channel(session);

    switch_event_create_subclass(&event, SWITCH_EVENT_CUSTOM, eventName);
    switch_channel_event_set_data(channel, event);
    if (json) {
        switch_event_add_body(event, "%s", json);
    }
    switch_event_fire(&event);
}

static switch_bool_t capture_callback(switch_media_bug_t *bug,
                                      void *user_data,
                                      switch_abc_type_t type)
{
    switch_core_session_t *session =
        switch_core_media_bug_get_session(bug);
    private_t *tech_pvt = (private_t *) user_data;

    switch (type) {

    case SWITCH_ABC_TYPE_INIT:
        break;

    case SWITCH_ABC_TYPE_CLOSE:
        {
            tech_pvt->close_requested = SWITCH_TRUE;

            int channel_closing = 1;

            stream_session_cleanup(session, NULL, channel_closing);
        }
        break;

    case SWITCH_ABC_TYPE_READ:
        if (tech_pvt->close_requested) {
            return SWITCH_FALSE;
        }
        return stream_frame(bug);

    case SWITCH_ABC_TYPE_WRITE_REPLACE:
        {
            switch_frame_t *frame =
                switch_core_media_bug_get_write_replace_frame(bug);

            if (!frame || !frame->data || frame->datalen == 0 || !tech_pvt || !tech_pvt->inject_buffer) {
                break;
            }

            switch_size_t need = frame->datalen;
            switch_size_t avail = 0;
            switch_size_t to_read = 0;
            switch_size_t got = 0;

            switch_mutex_lock(tech_pvt->mutex);

            if (!tech_pvt->inject_scratch || tech_pvt->inject_scratch_len < need) {
                uint8_t *nbuf = (uint8_t *)switch_core_session_alloc(session, need);
                if (!nbuf) {
                    switch_mutex_unlock(tech_pvt->mutex);
                    break;
                }
                tech_pvt->inject_scratch = nbuf;
                tech_pvt->inject_scratch_len = need;
            }

            uint8_t *inj = tech_pvt->inject_scratch;
            memset(inj, 0, need);

            avail = switch_buffer_inuse(tech_pvt->inject_buffer);

            /* Optional jitter buffer: wait until we have at least inject_min_buffer_ms queued. */
            if (tech_pvt->inject_min_buffer_ms > 0 && tech_pvt->inject_sample_rate > 0) {
                const switch_size_t bytes_per_ms = (switch_size_t)tech_pvt->inject_sample_rate * 2u * (switch_size_t)tech_pvt->channels / 1000u;
                const switch_size_t min_bytes = bytes_per_ms * (switch_size_t)tech_pvt->inject_min_buffer_ms;
                if (avail < min_bytes) {
                    /* Not enough audio yet: mix silence this frame to keep the call stable. */
                    to_read = 0;
                } else {
                    to_read = (avail > need) ? need : avail;
                }
            } else {
                to_read = (avail > need) ? need : avail;
            }
            if (to_read > 0) {
                got = switch_buffer_read(tech_pvt->inject_buffer, inj, to_read);
            }

            switch_mutex_unlock(tech_pvt->mutex);

            if (got < need) {
                tech_pvt->inject_underruns++;
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session),
                                  SWITCH_LOG_DEBUG,
                                  "(%s) PUSHBACK underrun: need=%lu got=%lu avail_before=%lu\n",
                                  switch_core_session_get_uuid(session),
                                  (unsigned long)need,
                                  (unsigned long)got,
                                  (unsigned long)avail);
            }

            {
                int16_t *dst = (int16_t *)frame->data;
                const int16_t *src = (const int16_t *)inj;
                const switch_size_t samples = need / 2;
                for (switch_size_t i = 0; i < samples; ++i) {
                    int32_t v = (int32_t)dst[i] + (int32_t)src[i];
                    if (v > 32767) v = 32767;
                    else if (v < -32768) v = -32768;
                    dst[i] = (int16_t)v;
                }
            }

            switch_core_media_bug_set_write_replace_frame(bug, frame);

            tech_pvt->inject_write_calls++;
            tech_pvt->inject_bytes += got;

            const switch_time_t now = switch_micro_time_now();
            if (!tech_pvt->inject_last_report) tech_pvt->inject_last_report = now;
            if ((now - tech_pvt->inject_last_report) > 1000000) {
                const double loss_pct = tech_pvt->inject_write_calls ?
                    (100.0 * (double)tech_pvt->inject_underruns / (double)tech_pvt->inject_write_calls) : 0.0;
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session),
                                  SWITCH_LOG_INFO,
                                  "(%s) PUSHBACK consume: write_calls=%llu bytes_read=%llu underruns=%llu loss%%=%.1f inject_inuse_now=%lu\n",
                                  switch_core_session_get_uuid(session),
                                  (unsigned long long)tech_pvt->inject_write_calls,
                                  (unsigned long long)tech_pvt->inject_bytes,
                                  (unsigned long long)tech_pvt->inject_underruns,
                                  loss_pct,
                                  (unsigned long)switch_buffer_inuse(tech_pvt->inject_buffer));
                tech_pvt->inject_last_report = now;
                tech_pvt->inject_write_calls = 0;
                tech_pvt->inject_bytes = 0;
                tech_pvt->inject_underruns = 0;
            }
        }
        break;

    default:
        break;
    }

    return SWITCH_TRUE;
}

static int get_channel_var_int(switch_core_session_t *session, const char* name, int def)
{
    switch_channel_t *channel = switch_core_session_get_channel(session);
    const char* val = channel ? switch_channel_get_variable(channel, name) : NULL;
    if (!val || !*val) return def;
    return atoi(val);
}

static switch_status_t start_capture(switch_core_session_t *session,
                                     switch_media_bug_flag_t flags,
                                     char* wsUri,
                                     int sampling,
                                     char* metadata)
{
    switch_channel_t *channel =
        switch_core_session_get_channel(session);

    switch_media_bug_t *bug;
    switch_codec_t *read_codec;
    void *pUserData = NULL;

    int channels = (flags & SMBF_STEREO) ? 2 : 1;
    /* Allow per-call tuning via channel vars (defaults handled in glue.cpp). */
    (void)get_channel_var_int(session, "STREAM_FRAME_MS", 0);

    if (switch_channel_get_private(channel, MY_BUG_NAME)) {
        return SWITCH_STATUS_FALSE;
    }

    if (switch_channel_pre_answer(channel)
        != SWITCH_STATUS_SUCCESS) {
        return SWITCH_STATUS_FALSE;
    }

    read_codec = switch_core_session_get_read_codec(session);

    if (stream_session_init(
            session,
            responseHandler,
            read_codec->implementation->actual_samples_per_second,
            wsUri,
            sampling,
            channels,
            metadata,
            &pUserData
        ) != SWITCH_STATUS_SUCCESS) {
        return SWITCH_STATUS_FALSE;
    }

    flags |= SMBF_READ_STREAM;
    flags |= SMBF_WRITE_REPLACE;

    if (switch_core_media_bug_add(
            session,
            MY_BUG_NAME,
            NULL,
            capture_callback,
            pUserData,
            0,
            flags,
            &bug
        ) != SWITCH_STATUS_SUCCESS) {
        return SWITCH_STATUS_FALSE;
    }

    switch_channel_set_private(channel, MY_BUG_NAME, bug);
    return SWITCH_STATUS_SUCCESS;
}

static switch_status_t do_stop(switch_core_session_t *session, char* text)
{
    return stream_session_cleanup(session, text, 0);
}

static switch_status_t do_pauseresume(switch_core_session_t *session, int pause)
{
    return stream_session_pauseresume(session, pause);
}

static switch_status_t send_text(switch_core_session_t *session, char* text)
{
    return stream_session_send_text(session, text);
}

#define STREAM_API_SYNTAX \
"<uuid> start <ws-uri> [mono|mixed|stereo] [8000|16000] [metadata]"

SWITCH_STANDARD_API(stream_function)
{
    char *argv[6] = { 0 };
    char *mycmd = NULL;
    int argc = 0;
    switch_status_t status = SWITCH_STATUS_FALSE;

    if (!zstr(cmd) && (mycmd = strdup(cmd))) {
        argc = switch_separate_string(mycmd, ' ', argv, 6);
    }

    if (argc < 2) {
        stream->write_function(stream, "-USAGE: %s\n", STREAM_API_SYNTAX);
        goto done;
    }

    switch_core_session_t *lsession =
        switch_core_session_locate(argv[0]);

    if (!lsession) goto done;

    if (!strcasecmp(argv[1], "stop")) {
        status = do_stop(lsession, argc > 2 ? argv[2] : NULL);
    }
    else if (!strcasecmp(argv[1], "pause")) {
        status = do_pauseresume(lsession, 1);
    }
    else if (!strcasecmp(argv[1], "resume")) {
        status = do_pauseresume(lsession, 0);
    }
    else if (!strcasecmp(argv[1], "send_text")) {
        status = (argc > 2 && argv[2]) ? send_text(lsession, argv[2]) : SWITCH_STATUS_FALSE;
    }
    else if (!strcasecmp(argv[1], "start")) {

    char wsUri[MAX_WS_URI];
    int sampling = 8000;
    switch_media_bug_flag_t flags = SMBF_READ_STREAM;
    int channels = 1;

        if (argc < 3 || !argv[2] || !validate_ws_uri(argv[2], wsUri)) {
            switch_core_session_rwunlock(lsession);
            goto done;
        }

        if (argc > 3 && argv[3]) {
            if (!strcasecmp(argv[3], "mono")) {
                channels = 1;
            } else if (!strcasecmp(argv[3], "mixed")) {
#ifdef SMBF_OPT_MIXED_READ
                flags |= SMBF_OPT_MIXED_READ;
#else
                /* Some FreeSWITCH builds don't expose SMBF_OPT_MIXED_READ; fall back to normal read stream. */
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(lsession), SWITCH_LOG_WARNING,
                                  "(%s) 'mixed' requested but SMBF_OPT_MIXED_READ not available; falling back to mono.\n",
                                  switch_core_session_get_uuid(lsession));
#endif
                channels = 1;
            } else if (!strcasecmp(argv[3], "stereo")) {
                flags |= SMBF_STEREO;
                channels = 2;
            }
        }

        if (argc > 4) sampling = atoi(argv[4]);

        status = start_capture(
            lsession,
            flags,
            wsUri,
            sampling,
            argc > 5 ? argv[5] : NULL
        );
    }

    switch_core_session_rwunlock(lsession);

done:
    if (status == SWITCH_STATUS_SUCCESS) {
        stream->write_function(stream, "+OK\n");
    } else {
        stream->write_function(stream, "-ERR\n");
    }

    switch_safe_free(mycmd);
    return SWITCH_STATUS_SUCCESS;
}

SWITCH_MODULE_LOAD_FUNCTION(mod_audio_stream_load)
{
    switch_api_interface_t *api_interface;

    *module_interface =
        switch_loadable_module_create_module_interface(pool, modname);

    switch_event_reserve_subclass(EVENT_JSON);
    switch_event_reserve_subclass(EVENT_CONNECT);
    switch_event_reserve_subclass(EVENT_ERROR);
    switch_event_reserve_subclass(EVENT_DISCONNECT);
    switch_event_reserve_subclass(EVENT_PLAY);

    SWITCH_ADD_API(
        api_interface,
        "uuid_audio_stream",
        "audio stream",
        stream_function,
        STREAM_API_SYNTAX
    );

    return SWITCH_STATUS_SUCCESS;
}

SWITCH_MODULE_SHUTDOWN_FUNCTION(mod_audio_stream_shutdown)
{
    switch_event_free_subclass(EVENT_JSON);
    switch_event_free_subclass(EVENT_CONNECT);
    switch_event_free_subclass(EVENT_DISCONNECT);
    switch_event_free_subclass(EVENT_ERROR);
    switch_event_free_subclass(EVENT_PLAY);
    return SWITCH_STATUS_SUCCESS;
}
