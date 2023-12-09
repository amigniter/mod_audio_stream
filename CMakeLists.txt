cmake_minimum_required(VERSION 3.18)
project(mod_audio_stream)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")

#set(ENV{PKG_CONFIG_PATH} "/usr/local/freeswitch/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(FreeSWITCH REQUIRED IMPORTED_TARGET freeswitch)
pkg_get_variable(FS_MOD_DIR freeswitch modulesdir)
message("FreeSWITCH modules dir: ${FS_MOD_DIR}")


add_library(mod_audio_stream SHARED mod_audio_stream.c mod_audio_stream.h audio_streamer_glue.h audio_streamer_glue.cpp streamer.cpp easywsclient.cpp tinythread.cpp buffer/ringbuffer.c)

set_property(TARGET mod_audio_stream PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(mod_audio_stream PRIVATE PkgConfig::FreeSWITCH mbedtls mbedx509 mbedcrypto)

install(TARGETS mod_audio_stream DESTINATION ${FS_MOD_DIR})