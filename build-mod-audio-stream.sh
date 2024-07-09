#!/bin/bash
### Initial one liner:
# sudo apt-get -y install git \
#    && cd /usr/src/ \
#    && git clone https://github.com/amigniter/mod_audio_stream.git \
#    && cd mod_audio_stream \
#    && sudo bash ./build-mod-audio-stream.sh

apt-get -y install libfreeswitch-dev libssl-dev zlib1g-dev libspeexdsp-dev

git submodule init
git submodule update

FS_PKGCONFIG=/usr/local/freeswitch/lib/pkgconfig
if [ -d "$FS_PKGCONFIG" ]; then
    export PKG_CONFIG_PATH=$FS_PKGCONFIG
fi

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
make install
