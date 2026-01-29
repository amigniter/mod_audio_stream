/*
 * mod_audio_stream FreeSWITCH module
 * READ  : FS → WebSocket (AI input)
 * WRITE : WebSocket → PCM → FS (AI output)
 */

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

/* ------------------------------------------------------------------ */
/* Event bridge                                                        */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/* Media bug callback                                                  */
/* ------------------------------------------------------------------ */
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
            int channel_closing =
                tech_pvt->close_requested ? 0 : 1;

            stream_session_cleanup(session, NULL, channel_closing);
        }
        break;

    /* ---------- FS → WS (READ) ---------- */
    case SWITCH_ABC_TYPE_READ:
        if (tech_pvt->close_requested) {
            return SWITCH_FALSE;
        }
        return stream_frame(bug);

    /* ---------- WS → FS (WRITE) ---------- */
    case SWITCH_ABC_TYPE_WRITE:
        {
            switch_frame_t *frame =
                switch_core_media_bug_get_write_replace_frame(bug);

            if (!frame || !tech_pvt->inject_buffer) {
                break;
            }

            /* How many bytes FS expects */
            switch_size_t need = frame->datalen;

            if (switch_buffer_inuse(tech_pvt->inject_buffer) < need) {
                /* Not enough AI audio yet → play silence */
                memset(frame->data, 0, frame->datalen);
                break;
            }

            /* Replace audio */
            switch_buffer_read(
                tech_pvt->inject_buffer,
                frame->data,
                need
            );
        }
        break;

    default:
        break;
    }

    return SWITCH_TRUE;
}

/* ------------------------------------------------------------------ */
/* Start capture                                                       */
/* ------------------------------------------------------------------ */
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

    /* Enable READ + WRITE replace */
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

/* ------------------------------------------------------------------ */
/* Stop / Pause / Resume                                               */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */
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
        status = send_text(lsession, argv[2]);
    }
    else if (!strcasecmp(argv[1], "start")) {

        char wsUri[MAX_WS_URI];
        int sampling = 8000;
        switch_media_bug_flag_t flags = SMBF_READ_STREAM;

        if (!validate_ws_uri(argv[2], wsUri)) goto done;

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

/* ------------------------------------------------------------------ */
/* Module load / unload                                                */
/* ------------------------------------------------------------------ */
SWITCH_MODULE_LOAD_FUNCTION(mod_audio_stream_load)
{
    switch_api_interface_t *api_interface;

    *module_interface =
        switch_loadable_module_create_module_interface(pool, modname);

    switch_event_reserve_subclass(EVENT_JSON);
    switch_event_reserve_subclass(EVENT_CONNECT);
    switch_event_reserve_subclass(EVENT_ERROR);
    switch_event_reserve_subclass(EVENT_DISCONNECT);

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
    return SWITCH_STATUS_SUCCESS;
}
