/*
 * mod_audio_stream FreeSWITCH module to stream audio to websocket and receive response
 */
#include <stdbool.h>
#include <math.h>
#include "mod_audio_stream.h"
#include "audio_streamer_glue.h"

SWITCH_MODULE_SHUTDOWN_FUNCTION(mod_audio_stream_shutdown);
SWITCH_MODULE_RUNTIME_FUNCTION(mod_audio_stream_runtime);
SWITCH_MODULE_LOAD_FUNCTION(mod_audio_stream_load);

SWITCH_MODULE_DEFINITION(mod_audio_stream, mod_audio_stream_load, mod_audio_stream_shutdown, NULL /*mod_audio_stream_runtime*/);

static void responseHandler(switch_core_session_t *session, const char *eventName, const char *json)
{
    switch_event_t *event;
    switch_channel_t *channel = switch_core_session_get_channel(session);
    switch_event_create_subclass(&event, SWITCH_EVENT_CUSTOM, eventName);
    switch_channel_event_set_data(channel, event);
    if (json)
        switch_event_add_body(event, "%s", json);
    switch_event_fire(&event);
}

static switch_bool_t capture_callback(switch_media_bug_t *bug, void *user_data, switch_abc_type_t type)
{
    switch_core_session_t *session = switch_core_media_bug_get_session(bug);

    switch (type)
    {
    case SWITCH_ABC_TYPE_INIT:
        break;

    case SWITCH_ABC_TYPE_CLOSE:
    {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "Got SWITCH_ABC_TYPE_CLOSE.\n");
        stream_session_cleanup(session, NULL, 1);
    }
    break;

    case SWITCH_ABC_TYPE_READ:
        return stream_frame(bug);
        break;

    case SWITCH_ABC_TYPE_WRITE:
    default:
        break;
    }

    return SWITCH_TRUE;
}

static switch_status_t start_capture(switch_core_session_t *session,
                                     switch_media_bug_flag_t flags,
                                     char *address,
                                     int sampling,
                                     char *metadata)
{
    switch_channel_t *channel = switch_core_session_get_channel(session);
    switch_media_bug_t *bug;
    switch_status_t status;
    switch_codec_t *read_codec;

    void *pUserData = NULL;
    int channels = (flags & SMBF_STEREO) ? 2 : 1;

    if (switch_channel_get_private(channel, MY_BUG_NAME))
    {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "mod_audio_stream: bug already attached!\n");
        return SWITCH_STATUS_FALSE;
    }

    read_codec = switch_core_session_get_read_codec(session);

    if (switch_channel_pre_answer(channel) != SWITCH_STATUS_SUCCESS)
    {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "mod_audio_stream: channel must have reached pre-answer status before calling start!\n");
        return SWITCH_STATUS_FALSE;
    }

    char wsUri[MAX_WS_URI];
    char tcpAddress[MAX_WS_URI];

    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "starting validate_address\n");
    int port = return_port(address);
    bool isWs = validate_address(address, wsUri, tcpAddress, 0);

    if (isWs)
    {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "calling stream_session_init for WS.\n");
        if (SWITCH_STATUS_FALSE == stream_session_init(session, responseHandler, read_codec->implementation->actual_samples_per_second,
                                                       wsUri, 0, sampling, channels, metadata, &pUserData))
        {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "Error initializing mod_audio_stream WS session.\n");
            return SWITCH_STATUS_FALSE;
        }
    }
    else
    {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "start_capture: port  = %i\n", port);
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "start_capture: address  = %s\n", address);
        address[strlen(address) - ((int)log10(port) + 2)] = '\0';
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "start_capture: address without port  = %s\n", address);
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "calling stream_session_init for TCP.\n");
        if (SWITCH_STATUS_FALSE == stream_session_init(session, responseHandler, read_codec->implementation->actual_samples_per_second,
                                                       address, port, sampling, channels, metadata, &pUserData))
        {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "Error initializing mod_audio_stream TCP session.\n");
            return SWITCH_STATUS_FALSE;
        }
    }

    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "adding bug.\n");
    if ((status = switch_core_media_bug_add(session, MY_BUG_NAME, NULL, capture_callback, pUserData, 0, flags, &bug)) != SWITCH_STATUS_SUCCESS)
    {
        return status;
    }
    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "setting bug private data.\n");
    switch_channel_set_private(channel, MY_BUG_NAME, bug);

    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "exiting start_capture.\n");
    return SWITCH_STATUS_SUCCESS;
}

static switch_status_t do_stop(switch_core_session_t *session, char *text)
{
    switch_status_t status = SWITCH_STATUS_SUCCESS;

    if (text)
    {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "mod_audio_stream: stop w/ final text %s\n", text);
    }
    else
    {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "mod_audio_stream: stop\n");
    }
    status = stream_session_cleanup(session, text, 0);

    return status;
}

static switch_status_t do_pauseresume(switch_core_session_t *session, int pause)
{
    switch_status_t status = SWITCH_STATUS_SUCCESS;

    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "mod_audio_stream: %s\n", pause ? "pause" : "resume");
    status = stream_session_pauseresume(session, pause);

    return status;
}

static switch_status_t send_text(switch_core_session_t *session, char *text)
{
    switch_status_t status = SWITCH_STATUS_FALSE;
    switch_channel_t *channel = switch_core_session_get_channel(session);
    switch_media_bug_t *bug = switch_channel_get_private(channel, MY_BUG_NAME);

    if (bug)
    {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "mod_audio_stream: sending text: %s.\n", text);
        status = stream_session_send_text(session, text);
    }
    else
    {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "mod_audio_stream: no bug, failed sending text: %s.\n", text);
    }
    return status;
}

#define STREAM_API_SYNTAX "<uuid> [start | stop | send_text | pause | resume | graceful-shutdown ] [wss-url | path] [mono | mixed | stereo] [8000 | 16000] [metadata]"
SWITCH_STANDARD_API(stream_function)
{
    char *mycmd = NULL, *argv[6] = {0};
    int argc = 0;

    switch_status_t status = SWITCH_STATUS_FALSE;

    if (!zstr(cmd) && (mycmd = strdup(cmd)))
    {
        argc = switch_separate_string(mycmd, ' ', argv, (sizeof(argv) / sizeof(argv[0])));
    }
    assert(cmd);
    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "mod_audio_stream cmd: %s\n", cmd ? cmd : "");

    if (zstr(cmd) || argc < 2 || (0 == strcmp(argv[1], "start") && argc < 4))
    {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "Error with command %s %s %s.\n", cmd, argv[0], argv[1]);
        stream->write_function(stream, "-USAGE: %s\n", STREAM_API_SYNTAX);
        goto done;
    }
    else
    {
        switch_core_session_t *lsession = NULL;
        if ((lsession = switch_core_session_locate(argv[0])))
        {
            if (!strcasecmp(argv[1], "stop"))
            {
                if (argc > 2 && (is_valid_utf8(argv[2]) != SWITCH_STATUS_SUCCESS))
                {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                                      "%s contains invalid utf8 characters\n", argv[2]);
                    switch_core_session_rwunlock(lsession);
                    goto done;
                }
                status = do_stop(lsession, argc > 2 ? argv[2] : NULL);
            }
            else if (!strcasecmp(argv[1], "pause"))
            {
                status = do_pauseresume(lsession, 1);
            }
            else if (!strcasecmp(argv[1], "resume"))
            {
                status = do_pauseresume(lsession, 0);
            }
            else if (!strcasecmp(argv[1], "send_text"))
            {
                if (argc < 3)
                {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                                      "send_text requires an argument specifying text to send\n");
                    switch_core_session_rwunlock(lsession);
                    goto done;
                }
                if (is_valid_utf8(argv[2]) != SWITCH_STATUS_SUCCESS)
                {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                                      "%s contains invalid utf8 characters\n", argv[2]);
                    switch_core_session_rwunlock(lsession);
                    goto done;
                }
                status = send_text(lsession, argv[2]);
            }
            else if (!strcasecmp(argv[1], "start"))
            {
                // switch_channel_t *channel = switch_core_session_get_channel(lsession);
                char address[MAX_WS_URI];
                int sampling = 8000;
                switch_media_bug_flag_t flags = SMBF_READ_STREAM;
                char *metadata = argc > 5 ? argv[5] : NULL;
                if (metadata && (is_valid_utf8(metadata) != SWITCH_STATUS_SUCCESS))
                {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                                      "%s contains invalid utf8 characters\n", metadata);
                    switch_core_session_rwunlock(lsession);
                    goto done;
                }
                if (0 == strcmp(argv[3], "mixed"))
                {
                    flags |= SMBF_WRITE_STREAM;
                }
                else if (0 == strcmp(argv[3], "stereo"))
                {
                    flags |= SMBF_WRITE_STREAM;
                    flags |= SMBF_STEREO;
                }
                else if (0 != strcmp(argv[3], "mono"))
                {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                                      "invalid mix type: %s, must be mono, mixed, or stereo\n", argv[3]);
                    switch_core_session_rwunlock(lsession);
                    goto done;
                }
                if (0 == strcmp(argv[4], "16k"))
                {
                    sampling = 16000;
                }
                else if (0 == strcmp(argv[4], "8k"))
                {
                    sampling = 8000;
                }
                else
                {
                    sampling = atoi(argv[4]);
                }
                if (strcmp(STREAM_TYPE, "WS") == 0 && !validate_address(argv[2], address, address, 0))
                {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "invalid address: %s\n", argv[2]);
                }
                else if (sampling % 8000 != 0)
                {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "invalid sample rate: %s\n", argv[4]);
                }
                else
                {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "starting start_capture\n");
                    status = start_capture(lsession, flags, argv[2], sampling, metadata);
                }
            }
            else
            {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                                  "unsupported mod_audio_stream cmd: %s\n", argv[1]);
            }
            switch_core_session_rwunlock(lsession);
        }
        else
        {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "Error locating session %s\n",
                              argv[0]);
        }
    }

    if (status == SWITCH_STATUS_SUCCESS)
    {
        stream->write_function(stream, "+OK Success\n");
    }
    else
    {
        stream->write_function(stream, "-ERR Operation Failed\n");
    }

done:
    switch_safe_free(mycmd);
    return SWITCH_STATUS_SUCCESS;
}

SWITCH_MODULE_LOAD_FUNCTION(mod_audio_stream_load)
{
    switch_api_interface_t *api_interface;

    switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_NOTICE, "mod_audio_stream API loading..\n");

    /* connect my internal structure to the blank pointer passed to me */
    *module_interface = switch_loadable_module_create_module_interface(pool, modname);

    /* create/register custom event message types */
    if (switch_event_reserve_subclass(EVENT_JSON) != SWITCH_STATUS_SUCCESS ||
        switch_event_reserve_subclass(EVENT_CONNECT) != SWITCH_STATUS_SUCCESS ||
        switch_event_reserve_subclass(EVENT_ERROR) != SWITCH_STATUS_SUCCESS ||
        switch_event_reserve_subclass(EVENT_DISCONNECT) != SWITCH_STATUS_SUCCESS)
    {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR, "Couldn't register an event subclass for mod_audio_stream API.\n");
        return SWITCH_STATUS_TERM;
    }
    SWITCH_ADD_API(api_interface, "uuid_audio_stream", "audio_stream API", stream_function, STREAM_API_SYNTAX);
    switch_console_set_complete("add uuid_audio_stream start wss-url metadata");
    switch_console_set_complete("add uuid_audio_stream start wss-url");
    switch_console_set_complete("add uuid_audio_stream stop");

    switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_NOTICE, "mod_audio_stream API successfully loaded\n");

    /* indicate that the module should continue to be loaded */
    return SWITCH_STATUS_SUCCESS;
}

/*
  Called when the system shuts down
  Macro expands to: switch_status_t mod_audio_stream_shutdown() */
SWITCH_MODULE_SHUTDOWN_FUNCTION(mod_audio_stream_shutdown)
{
    switch_event_free_subclass(EVENT_JSON);
    switch_event_free_subclass(EVENT_CONNECT);
    switch_event_free_subclass(EVENT_DISCONNECT);
    switch_event_free_subclass(EVENT_ERROR);

    return SWITCH_STATUS_SUCCESS;
}
