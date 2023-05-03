cmake_minimum_required(VERSION 3.18)
project(mod_audio_stream)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")

find_package(PkgConfig REQUIRED)
pkg_check_modules(FreeSWITCH REQUIRED IMPORTED_TARGET freeswitch)
pkg_get_variable(FS_MOD_DIR freeswitch modulesdir)
message("FreeSWITCH modules dir: ${FS_MOD_DIR}")

set(USE_TLS ON CACHE BOOL "Use TLS for secure WebSocket connections" FORCE)
add_subdirectory(libs/IXWebSocket)

if(CMAKE_BUILD_TYPE MATCHES "Release")
    # Add linker flags to strip symbols and reduce size of static library
    set_target_properties(ixwebsocket PROPERTIES LINK_FLAGS_RELEASE "-s -w") #-static-libgcc -static-libstdc++
endif()

add_library(mod_audio_stream SHARED mod_audio_stream.c mod_audio_stream.h audio_streamer_glue.h audio_streamer_glue.cpp)

set_property(TARGET mod_audio_stream PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(mod_audio_stream PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs/IXWebSocket)

target_link_libraries(mod_audio_stream PRIVATE PkgConfig::FreeSWITCH pthread)
target_link_libraries (mod_audio_stream PRIVATE ixwebsocket)

install(TARGETS mod_audio_stream DESTINATION ${FS_MOD_DIR})