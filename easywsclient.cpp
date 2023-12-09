#include "easywsclient.hpp"

#include <mbedtls/entropy.h>
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/net_sockets.h>
#include <mbedtls/error.h>
#include <mbedtls/platform.h>
//#include <mbedtls/debug.h>
#include <cassert>
#include <cstdio>
#include <cstring>

#include "tinythread.h"

#if !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0501)
#undef _WIN32_WINNT
/* Enables getaddrinfo() & Co */
#define _WIN32_WINNT 0x0501
#endif

#if _MSC_VER >= 1700
#   ifndef snprintf
#       define snprintf _snprintf_s
#   endif
#endif /* WIN32 */


namespace { // private module-only namespace

    class BaseSocket
    {
        std::string error_string;

    protected:
        BaseSocket()
        {
            // reserve space for the error message, because marmalades memory manager is not thread-safe
            error_string.reserve(512);
        }

        void set_errstr(int res)
        {
            char buf[256];
            mbedtls_strerror(res, buf, sizeof(buf));
            error_string = buf;
        }

        void set_errstr(std::string e)
        {
            error_string = e;
        }

    public:
        virtual bool connect(const char *host, short port) = 0;
        virtual void close() = 0;
        virtual void set_blocking(bool should_block) = 0;

        virtual int send(const char *buf, size_t siz) = 0;
        virtual int recv(char *buf, size_t siz) = 0;

        std::string get_error_string() { return error_string; }
        virtual ~BaseSocket() {}
    };

    class TCPSocket : public BaseSocket
    {
    private:
        bool is_connected;
    protected:
        mbedtls_net_context net;
    public:
        TCPSocket():is_connected(false)
        {
            mbedtls_net_init(&net);
        }

        virtual ~TCPSocket()
        {
            mbedtls_net_free(&net);
        }

        virtual bool connect(const char *host, short port)
        {
            if (is_connected)
                return true;

            char port_str[8];
            snprintf(port_str, 8, "%hu", port);

            int res = mbedtls_net_connect(&net, host, port_str, MBEDTLS_NET_PROTO_TCP);
            if (res != 0)
            {
                set_errstr(res);
                close();
                return false;
            }

            is_connected = true;
            return true;
        }

        virtual void close()
        {
            is_connected = false;
        }

        virtual int send(const char *buf, size_t siz)
        {
            return mbedtls_net_send(&net, (unsigned char *)buf, siz);
        }

        virtual int recv(char *buf, size_t siz)
        {
            return mbedtls_net_recv(&net, (unsigned char *)buf, siz);
        }

        virtual void set_blocking(bool should_block)
        {
            assert(is_connected);

            if (should_block)
                mbedtls_net_set_block(&net);
            else
                mbedtls_net_set_nonblock(&net);
        }
    };


    class TLSSocket : public TCPSocket
    {
    private:
        bool is_connected; // TLSSocket is_connected is different (private) than TCPSocket is_connected
        // here it indicates whether SSL initialization and handshake has succeded

        mbedtls_ssl_context ssl;
        mbedtls_ssl_config conf;
        mbedtls_entropy_context entropy;
        mbedtls_ctr_drbg_context ctr_drbg;
        //mbedtls_x509_crt cacert;

        static void debug_print(void *ctx, int level, const char *file, int line, const char *str)
        {
            ((void)level);
            mbedtls_fprintf((FILE *)ctx, "%s:%04d: %s", file, line, str);
            //fflush((FILE *)ctx);
        }
    public:
        TLSSocket():is_connected(false)
        {
            //mbedtls_debug_set_threshold( 1000 );
            mbedtls_ssl_init(&ssl);
            mbedtls_ssl_config_init(&conf);

            //mbedtls_x509_crt_init( &cacert );
            mbedtls_ctr_drbg_init(&ctr_drbg);
            mbedtls_entropy_init(&entropy);
        }

        virtual ~TLSSocket()
        {
            //mbedtls_x509_crt_free( &cacert );
            mbedtls_ssl_free(&ssl);
            mbedtls_ssl_config_free(&conf);
            mbedtls_ctr_drbg_free(&ctr_drbg);
            mbedtls_entropy_free(&entropy);
        }

        virtual bool connect(const char *host, short port)
        {
            if (is_connected)
                return true;

            if (!TCPSocket::connect(host, port))
            {
                return false;
            }

            int res = mbedtls_ctr_drbg_seed(&ctr_drbg, mbedtls_entropy_func, &entropy, nullptr, 0);
            res = mbedtls_ssl_config_defaults(&conf,
                                              MBEDTLS_SSL_IS_CLIENT,
                                              MBEDTLS_SSL_TRANSPORT_STREAM,
                                              MBEDTLS_SSL_PRESET_DEFAULT);

            if (res != 0)
            {
                set_errstr(res);
                close();
                return false;
            }

            /* OPTIONAL is not optimal for security,
            * but makes interop easier in this simplified example */
            //mbedtls_ssl_conf_authmode(&conf, MBEDTLS_SSL_VERIFY_REQUIRED);
            mbedtls_ssl_conf_authmode(&conf, MBEDTLS_SSL_VERIFY_NONE);

            //ret = mbedtls_x509_crt_parse( &cacert, (const unsigned char *) mbedtls_test_cas_pem, mbedtls_test_cas_pem_len );
            //mbedtls_ssl_conf_ca_chain( &conf, &cacert, NULL );

            mbedtls_ssl_conf_rng(&conf, mbedtls_ctr_drbg_random, &ctr_drbg);
            mbedtls_ssl_conf_dbg(&conf, debug_print, stderr);

            res = mbedtls_ssl_setup(&ssl, &conf);
            if (res != 0)
            {
                set_errstr(res);
                close();
                return false;
            }

            res = mbedtls_ssl_set_hostname(&ssl, host);
            if (res != 0)
            {
                set_errstr(res);
                close();
                return false;
            }

            mbedtls_ssl_set_bio(&ssl, &net, mbedtls_net_send, mbedtls_net_recv, 0);// , mbedtls_net_recv_timeout);

            do res = mbedtls_ssl_handshake(&ssl);
            while (res == MBEDTLS_ERR_SSL_WANT_READ || res == MBEDTLS_ERR_SSL_WANT_WRITE);

            if (res != 0)
            {
                set_errstr(res);
                close();
                return false;
            }

            uint32_t flags;
            if ((flags = mbedtls_ssl_get_verify_result(&ssl)) != 0)
            {
                char vrfy_buf[512];
                mbedtls_x509_crt_verify_info(vrfy_buf, sizeof(vrfy_buf), "  ! ", flags);
                set_errstr(vrfy_buf);
                close();
                return false;
            }

            is_connected = true;
            return true;
        }

        virtual void close()
        {
            TCPSocket::close();
            is_connected = false;
        }

        virtual int send(const char *buf, size_t siz)
        {
            return mbedtls_ssl_write(&ssl, (unsigned char *)buf, siz);
            /*int ret = 0;
            do ret = mbedtls_ssl_write(&ssl, (unsigned char *)buf, siz);
            while (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE);
            if (ret < 0) set_errstr(ret);
            return ret;*/
        }

        virtual int recv(char *buf, size_t siz)
        {
            return mbedtls_ssl_read(&ssl, (unsigned char *)buf, siz);
            /*int ret = 0;
            do ret = mbedtls_ssl_read(&ssl, (unsigned char *)buf, siz);
            while (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE);
            if (ret < 0) set_errstr(ret);
            return ret;*/
        }
    };

    class _RealWebSocket : public easywsclient::WebSocket
    {
    public:
        // http://tools.ietf.org/html/rfc6455#section-5.2  Base Framing Protocol
        //
        //  0                   1                   2                   3
        //  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
        // +-+-+-+-+-------+-+-------------+-------------------------------+
        // |F|R|R|R| opcode|M| Payload len |    Extended payload length    |
        // |I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
        // |N|V|V|V|       |S|             |   (if payload len==126/127)   |
        // | |1|2|3|       |K|             |                               |
        // +-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
        // |     Extended payload length continued, if payload len == 127  |
        // + - - - - - - - - - - - - - - - +-------------------------------+
        // |                               |Masking-key, if MASK set to 1  |
        // +-------------------------------+-------------------------------+
        // | Masking-key (continued)       |          Payload Data         |
        // +-------------------------------- - - - - - - - - - - - - - - - +
        // :                     Payload Data continued ...                :
        // + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
        // |                     Payload Data continued ...                |
        // +---------------------------------------------------------------+
        struct wsheader_type {
            unsigned header_size;
            bool fin;
            bool mask;
            enum opcode_type {
                CONTINUATION = 0x0,
                TEXT_FRAME = 0x1,
                BINARY_FRAME = 0x2,
                CLOSE = 8,
                PING = 9,
                PONG = 0xa
            } opcode;
            int N0;
            uint64_t N;
            uint8_t masking_key[4];
        };

        std::vector<char> rxbuf;
        std::vector<char> txbuf;

        readyStateValues readyState;
        bool useMask;
        BaseSocket* socket;
        tthread::mutex write_buffer_mutex;

        _RealWebSocket(std::string host, std::string path, int port, std::string url, std::string origin, bool _useMask, bool _useSSL)
        {
            m_host = host;
            m_path = path;
            m_port = port;
            m_url = url;
            m_origin = origin;

            useMask = _useMask;

            if (_useSSL)
                socket = new TLSSocket();
            else
                socket = new TCPSocket();

            readyState = CONNECTING;
            ipLookup = keNone;
        }

        virtual ~_RealWebSocket()
        {
            delete socket;
        }

        readyStateValues getReadyState() const {
            return readyState;
        }

        void poll(int timeout, WSErrorCallback errorCallback, void* userData)  // timeout in milliseconds
        {
            (void)timeout; //unused

            using namespace easywsclient;

            if(readyState == CONNECTING)
            {
                if(ipLookup == keComplete)
                {
                    assert(socket);

                    // if TCP socket is already connected it will just go to SSL handshake
                    // it is because TCPSocket::is_connected is already true, but TLSSocket::is_connected is false

                    // establish the ssl connection and do websocket handshaking
                    if (!socket->connect(m_host.c_str(), static_cast<short>(m_port)) || !doConnect2(errorCallback, userData))
                    {
                        forceClose();
                    }
                    else
                    {
                        readyState = OPEN;
                    }
                }
                else if( ipLookup == keFailed )
                {
                    forceClose();
                    using namespace easywsclient;
                    errorCallback(WSError(WSError::DNS_LOOKUP_FAILED, "DNS Lookup failed"), userData);
                }
                else if ( ipLookup == keNone)
                {
                    // initial tcp connect
                    connect();
                }
            }
            else if(ipLookup == keComplete)
            {
                assert(timeout == 0); // not implemented yet: use mbedtls_net_recv_timeout et. all.

                if (readyState == CLOSED)
                {
                    return;
                }

                using namespace easywsclient;

                for(;;) // while(true), but without a warning about constant expression
                {
                    // FD_ISSET(0, &rfds) will be true
                    std::vector<char>::size_type N = rxbuf.size();
                    rxbuf.resize(N + 1500);

                    assert(socket);

                    int ret = socket->recv(&rxbuf[0] + N, 1500);

                    if (ret < 0 && (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE))
                    {
                        rxbuf.resize(N);
                        break;
                    }
                    else if (ret <= 0)
                    {
                        rxbuf.resize(N);
                        socket->close();
                        readyState = CLOSED;
                        if (ret < 0)
                        {
                            fputs("Connection error!\n", stderr);
                            errorCallback(WSError(WSError::RECV_FAILED, "recv or SSL_read failed"), userData);
                        }
                        else
                        {
                            fputs("Connection closed!\n", stderr);
                            errorCallback(WSError(WSError::CONNECTION_CLOSED, "Connection closed"), userData);
                        }
                        break;
                    }
                    else
                    {
                        rxbuf.resize(static_cast<std::vector<char>::size_type>(N + ret));
                    }
                }
                write_buffer_mutex.lock();
                while (txbuf.size())
                {
                    assert(socket);

                    int ret = socket->send(&txbuf[0], txbuf.size());

                    if (ret < 0 && (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE))
                    {
                        break;
                    }
                    else if (ret <= 0)
                    {
                        socket->close();
                        readyState = CLOSED;
                        if (ret < 0)
                        {
                            fputs("Connection error!\n", stderr);
                            errorCallback(WSError(WSError::SEND_FAILED, "send or SSL_write failed"), userData);
                        }
                        else
                        {
                            fputs("Connection closed!\n", stderr);
                            errorCallback(WSError(WSError::CONNECTION_CLOSED, "Connection closed"), userData);
                        }
                        break;
                    }
                    else
                    {
                        assert(ret <= (int)txbuf.size());
                        txbuf.erase(txbuf.begin(), txbuf.begin() + ret);
                    }
                }
                write_buffer_mutex.unlock();
            }

            write_buffer_mutex.lock();
            if (!txbuf.size() && readyState == CLOSING)
            {
                socket->close();
                readyState = CLOSED;
                errorCallback(WSError(WSError::CONNECTION_CLOSED, "Connection closed"), userData);
            }
            write_buffer_mutex.unlock();
        }

        // Callable must have signature: void(const std::string & message).
        // Should work with C functions, C++ functors, and C++11 std::function and
        // lambda:
        //template<class Callable>
        //void dispatch(Callable callable)
        virtual void _dispatch(WSMessageCallback messageCallback, WSErrorCallback errorCallback, void* userData)
        {
            if(readyState == CONNECTING) return;

            // TODO: consider acquiring a lock on rxbuf...
            for(;;) // while (true) withoput warning about constant expression
            {

                wsheader_type ws;
                {
                    if (rxbuf.size() < 2) { return; /* Need at least 2 */ }
                    const uint8_t * data = (uint8_t *) &rxbuf[0]; // peek, but don't consume
                    ws.fin = (data[0] & 0x80) == 0x80;
                    ws.opcode = (wsheader_type::opcode_type) (data[0] & 0x0f);
                    ws.mask = (data[1] & 0x80) == 0x80;
                    ws.N0 = (data[1] & 0x7f);
                    ws.header_size = 2 + (ws.N0 == 126? 2 : 0) + (ws.N0 == 127? 8 : 0) + (ws.mask? 4 : 0);
                    if (rxbuf.size() < ws.header_size) { return; /* Need: ws.header_size - rxbuf.size() */ }
                    int data_offset = -1;
                    if (ws.N0 < 126) {
                        ws.N = ws.N0;
                        data_offset = 2;
                    }
                    else if (ws.N0 == 126) {
                        ws.N = 0;
                        ws.N |= ((uint64_t) data[2]) << 8;
                        ws.N |= ((uint64_t) data[3]) << 0;
                        data_offset = 4;
                    }
                    else if (ws.N0 == 127) {
                        ws.N = 0;
                        ws.N |= ((uint64_t) data[2]) << 56;
                        ws.N |= ((uint64_t) data[3]) << 48;
                        ws.N |= ((uint64_t) data[4]) << 40;
                        ws.N |= ((uint64_t) data[5]) << 32;
                        ws.N |= ((uint64_t) data[6]) << 24;
                        ws.N |= ((uint64_t) data[7]) << 16;
                        ws.N |= ((uint64_t) data[8]) << 8;
                        ws.N |= ((uint64_t) data[9]) << 0;
                        data_offset = 10;
                    }
                    if (ws.mask) {
                        assert(data_offset != -1);
                        ws.masking_key[0] = ((uint8_t) data[data_offset+0]) << 0;
                        ws.masking_key[1] = ((uint8_t) data[data_offset+1]) << 0;
                        ws.masking_key[2] = ((uint8_t) data[data_offset+2]) << 0;
                        ws.masking_key[3] = ((uint8_t) data[data_offset+3]) << 0;
                    }
                    else {
                        ws.masking_key[0] = 0;
                        ws.masking_key[1] = 0;
                        ws.masking_key[2] = 0;
                        ws.masking_key[3] = 0;
                    }
                    if (rxbuf.size() < ws.header_size+ws.N) { return; /* Need: ws.header_size+ws.N - rxbuf.size() */ }
                }


                // We got a whole message, now do something with it:
                if (ws.opcode == wsheader_type::TEXT_FRAME && ws.fin) {
                    if (ws.mask) { for (size_t i = 0; i != ws.N; ++i) { rxbuf[static_cast<std::vector<char>::size_type>(i+ws.header_size)] ^= ws.masking_key[i&0x3]; } }
                    std::string data(rxbuf.begin()+ws.header_size, rxbuf.begin()+ws.header_size+(size_t)ws.N);
                    messageCallback((const std::string) data, userData);
                }
                else if (ws.opcode == wsheader_type::PING) {
                    if (ws.mask) { for (size_t i = 0; i != ws.N; ++i) { rxbuf[static_cast<std::vector<char>::size_type>(i+ws.header_size)] ^= ws.masking_key[i&0x3]; } }
                    std::string data(rxbuf.begin()+ws.header_size, rxbuf.begin()+ws.header_size+(size_t)ws.N);
                    sendData(wsheader_type::PONG, data.size(), data.begin(), data.end());
                }
                else if (ws.opcode == wsheader_type::PONG)
                {
                    messageCallback((const std::string) "{ \"@class\" : \".pong\" }", userData);
                }
                else if (ws.opcode == wsheader_type::CLOSE)
                {
                    close();
                }
                else
                {
                    fprintf(stderr, "ERROR: Got unexpected WebSocket message.\n");
                    using namespace easywsclient;
                    errorCallback(WSError(WSError::UNEXPECTED_MESSAGE, "Got unexpected WebSocket message."), userData);
                    close();
                }

                rxbuf.erase(rxbuf.begin(), rxbuf.begin() + ws.header_size+(size_t)ws.N);
            }
        }

        void sendPing()
        {
            if(readyState == CONNECTING) return;
            std::string empty;
            sendData(wsheader_type::PING, empty.size(), empty.begin(), empty.end());
        }

        void send(const std::string& message)
        {
            if(readyState == CONNECTING) return;
            sendData(wsheader_type::TEXT_FRAME, message.size(), message.begin(), message.end());
        }

        void sendBinary(const std::vector<uint8_t>& message) {
            if(readyState == CONNECTING) return;
            sendData(wsheader_type::BINARY_FRAME, message.size(), message.begin(), message.end());
        }

        template<class Iterator>
        void sendData(wsheader_type::opcode_type type, uint64_t message_size, Iterator message_begin, Iterator message_end)
        {
            // TODO:
            // Masking key should (must) be derived from a high quality random
            // number generator, to mitigate attacks on non-WebSocket friendly
            // middleware:
            const uint8_t masking_key[4] = { 0x12, 0x34, 0x56, 0x78 };
            // TODO: consider acquiring a lock on txbuf...
            if (readyState == CLOSING || readyState == CLOSED || readyState == CONNECTING) { return; }
            std::vector<uint8_t> header;
            //uint64_t message_size = message.size();
            header.assign(2 + (message_size >= 126 ? 2 : 0) + (message_size >= 65536 ? 6 : 0) + (useMask ? 4 : 0), 0);
            header[0] = uint8_t(0x80 | type);

            if (message_size < 126) {
                header[1] = (message_size & 0xff) | (useMask ? 0x80 : 0);
                if (useMask) {
                    header[2] = masking_key[0];
                    header[3] = masking_key[1];
                    header[4] = masking_key[2];
                    header[5] = masking_key[3];
                }
            }
            else if (message_size < 65536) {
                header[1] = 126 | (useMask ? 0x80 : 0);
                header[2] = (message_size >> 8) & 0xff;
                header[3] = (message_size >> 0) & 0xff;
                if (useMask) {
                    header[4] = masking_key[0];
                    header[5] = masking_key[1];
                    header[6] = masking_key[2];
                    header[7] = masking_key[3];
                }
            }
            else { // TODO: run coverage testing here
                header[1] = 127 | (useMask ? 0x80 : 0);
                header[2] = (message_size >> 56) & 0xff;
                header[3] = (message_size >> 48) & 0xff;
                header[4] = (message_size >> 40) & 0xff;
                header[5] = (message_size >> 32) & 0xff;
                header[6] = (message_size >> 24) & 0xff;
                header[7] = (message_size >> 16) & 0xff;
                header[8] = (message_size >>  8) & 0xff;
                header[9] = (message_size >>  0) & 0xff;
                if (useMask) {
                    header[10] = masking_key[0];
                    header[11] = masking_key[1];
                    header[12] = masking_key[2];
                    header[13] = masking_key[3];
                }
            }
            write_buffer_mutex.lock();
            // N.B. - txbuf will keep growing until it can be transmitted over the socket:
            txbuf.insert(txbuf.end(), header.begin(), header.end());
            txbuf.insert(txbuf.end(), message_begin, message_end);
            if (useMask) {
                for (size_t i = 0; i != message_size; ++i) { *(txbuf.end() - message_size + i) ^= masking_key[i&0x3]; }
                /*size_t message_offset = txbuf.size() - message_size;
                for (size_t i = 0; i != message_size; ++i) {
                    txbuf[message_offset + i] ^= masking_key[i&0x3];
                }*/
            }
            write_buffer_mutex.unlock();
        }

        void close() {
            if(readyState == CLOSING || readyState == CLOSED) { return; }
            readyState = CLOSING;
            uint8_t closeFrame[6] = {0x88, 0x80, 0x00, 0x00, 0x00, 0x00}; // last 4 bytes are a masking key
            std::vector<uint8_t> header(closeFrame, closeFrame+6);
            write_buffer_mutex.lock();
            txbuf.insert(txbuf.end(), header.begin(), header.end());
            write_buffer_mutex.unlock();
        }

        void forceClose()
        {
            if(readyState == CLOSING || readyState == CLOSED) { return; }
            readyState = CLOSING;
            txbuf.clear();
        }

        void connect() // change vs GameSparks implementation
        {
            readyState = CONNECTING;
            ipLookup = keInprogress;
            assert(socket);

            if (!((TCPSocket*)(socket))->TCPSocket::connect(m_host.c_str(), static_cast<short>(m_port)))
            {
                ipLookup = keFailed;
            }
            else
            {
                ipLookup = keComplete;
            }

        }

        bool doConnect2(WSErrorCallback errorCallback, void* userData)
        {
            using namespace easywsclient;

#define SEND(buf)  socket->send(buf, strlen(buf))
#define RECV(buf)  socket->recv(buf, 1)

            {
                // XXX: this should be done non-blocking,
                char line[256];
                int status;
                int i;
                snprintf(line, 256, "GET /%s HTTP/1.1\r\n", m_path.c_str()); SEND(line);
                if (m_port == 80) {
                    snprintf(line, 256, "Host: %s\r\n", m_host.c_str()); SEND(line);
                }
                else {
                    snprintf(line, 256, "Host: %s:%d\r\n", m_host.c_str(), m_port); SEND(line);
                }
                snprintf(line, 256, "Authorization: 15db07114504480519240fcc892fcd25e357cedf\r\n"); SEND(line);
                snprintf(line, 256, "Upgrade: websocket\r\n"); SEND(line);
                snprintf(line, 256, "Connection: Upgrade\r\n"); SEND(line);
                if (!m_origin.empty()) {
                    snprintf(line, 256, "Origin: %s\r\n", m_origin.c_str()); SEND(line);
                }
                snprintf(line, 256, "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==\r\n"); SEND(line);
                snprintf(line, 256, "Sec-WebSocket-Version: 13\r\n"); SEND(line);
                snprintf(line, 256, "\r\n"); SEND(line);
                for (i = 0; i < 2 || (i < 255 && line[i-2] != '\r' && line[i-1] != '\n'); ++i)
                {
                    if (RECV(line+i) == 0)
                    {
                        errorCallback(WSError(WSError::CLOSED_DURING_WS_HANDSHAKE, "The connection was closed while the websocket handshake was in progress."), userData);
                        return false;
                    }
                }
                line[i] = 0;
                if (i == 255)
                {
                    fprintf(stderr, "ERROR: Got invalid status line connecting to: %s\n", m_url.c_str());
                    errorCallback(WSError(WSError::INVALID_STATUS_LINE_DURING_WS_HANDSHAKE, "Got invalid status line connecting to : " + m_url), userData);
                    return false;
                }
                if (sscanf(line, "HTTP/1.1 %d", &status) != 1 || status != 101)
                {
                    fprintf(stderr, "ERROR: Got bad status connecting to %s: %s", m_url.c_str(), line);
                    errorCallback(WSError(WSError::BAD_STATUS_CODE, "Got bad status connecting to : " + m_url), userData);
                    return false;
                }
                // TODO: verify response headers,
                for(;;) // while (true)
                {
                    for (i = 0; i < 2 || (i < 255 && line[i-2] != '\r' && line[i-1] != '\n'); ++i)
                    {
                        if (RECV(line+i) == 0)
                        {
                            errorCallback(WSError(WSError::CLOSED_DURING_WS_HANDSHAKE, "The connection was closed while the websocket handshake was in progress."), userData);
                            return false;
                        }
                    }
                    if (line[0] == '\r' && line[1] == '\n')
                    {
                        break;
                    }
                }
            }

            socket->set_blocking(false);
            fprintf(stderr, "Connected to: %s\n", m_url.c_str());

            return true;
        }
    };

    easywsclient::WebSocket::pointer from_url(const std::string& url, bool useMask, const std::string& origin)
    {
        char host[256];
        int port;
        char path[256];

        bool secure_connection = false;

        if (url.size() >= 256) {
            fprintf(stderr, "ERROR: url size limit exceeded: %s\n", url.c_str());
            return NULL;
        }
        if (origin.size() >= 200) {
            fprintf(stderr, "ERROR: origin size limit exceeded: %s\n", origin.c_str());
            return NULL;
        }

        if (sscanf(url.c_str(), "ws://%[^:/]:%d/%s", host, &port, path) == 3) {
        }
        else if (sscanf(url.c_str(), "ws://%[^:/]/%s", host, path) == 2) {
            port = 80;
        }
        else if (sscanf(url.c_str(), "ws://%[^:/]:%d", host, &port) == 2) {
            path[0] = '\0';
        }
        else if (sscanf(url.c_str(), "ws://%[^:/]", host) == 1) {
            port = 80;
            path[0] = '\0';
        }
        else if (sscanf(url.c_str(), "wss://%[^:/]:%d/%s", host, &port, path) == 3) {
            secure_connection = true;
        }
        else if (sscanf(url.c_str(), "wss://%[^:/]/%s", host, path) == 2) {
            port = 443;
            secure_connection = true;
        }
        else if (sscanf(url.c_str(), "wss://%[^:/]:%d", host, &port) == 2) {
            path[0] = '\0';
            secure_connection = true;
        }
        else if (sscanf(url.c_str(), "wss://%[^:/]", host) == 1) {
            port = 443;
            path[0] = '\0';
            secure_connection = true;
        }
        else
        {
            fprintf(stderr, "ERROR: Could not parse WebSocket url: %s\n", url.c_str());
            return NULL;
        }
        fprintf(stderr, "easywsclient: connecting: host=%s port=%d path=/%s\n", host, port, path);

        _RealWebSocket *nWebsocket = new _RealWebSocket(host, path, port, url, origin, useMask, secure_connection);

        return easywsclient::WebSocket::pointer(nWebsocket);
    }
} // end of module-only namespace



namespace easywsclient {
    WebSocket::pointer WebSocket::from_url(const std::string& url, const std::string& origin) {
        return ::from_url(url, true, origin);
    }

    WebSocket::pointer WebSocket::from_url_no_mask(const std::string& url, const std::string& origin) {
        return ::from_url(url, false, origin);
    }
} // namespace easywsclient
