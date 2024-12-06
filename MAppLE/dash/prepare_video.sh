#!/bin/sh

SRC="sample.mp4"
SRCURL="https://www.pexels.com/download/video/3195394/"

if [ ! -f "$SRC" ]; then
    wget $SRCURL -O $SRC_ZIP
fi

function encode {
    mkdir -p video$1s
    HAS_AUDIO=$(ffprobe -i "$SRC" -show_streams -select_streams a -loglevel error)
    if [ -n "$HAS_AUDIO" ]; then
        echo "Audio stream detected. Including in DASH output."
        ffmpeg -re -stream_loop 4 -i "$SRC" -c:v libx264 -preset ultrafast -c:a aac \
            -map 0:v:0 -b:v:0 10M -maxrate:v:0 10M -s:v:0 2560x1440 -profile:v:0 baseline \
            -map 0:a:0 -b:a 128k \
            -bufsize 0.5M -bf 1 -keyint_min 4 -g 5 -sc_threshold 0 \
            -b_strategy 0 -use_template 1 -use_timeline 1 \
            -seg_duration $1 -streaming 1 \
            -adaptation_sets "id=0,streams=v id=1,streams=a" \
            -f dash video$1s/manifest.mpd
    else
        echo "No audio stream detected. Creating video-only DASH output."
        ffmpeg -re -stream_loop 4 -i "$SRC" -c:v libx264 -preset ultrafast \
            -map 0:v:0 -b:v:0 10M -maxrate:v:0 10M -s:v:0 2560x1440 -profile:v:0 baseline \
            -bufsize 0.5M -bf 1 -keyint_min 4 -g 5 -sc_threshold 0 \
            -b_strategy 0 -use_template 1 -use_timeline 1 \
            -seg_duration $1 -streaming 1 \
            -adaptation_sets "id=0,streams=v" \
            -f dash video$1s/manifest.mpd
    fi


    ls -l video/*stream0* | awk '{print $5}' > chunksizes-stream0.txt
    ls -l video/*stream1* | awk '{print $5}' > chunksizes-stream1.txt
    ls -l video/*stream2* | awk '{print $5}' > chunksizes-stream2.txt
}

# encode 0.1
# encode 0.2
# encode 0.3
# encode 0.4
# encode 0.5
# encode 0.6
# encode 0.7
# encode 0.8
# encode 0.9
# encode 1
# encode 2
# encode 3
# encode 4
encode 5

