#!/bin/sh

set -e

# Set up display; otherwise rendering will fail
Xvfb -screen 0 320x240x24 &
export DISPLAY=:0

# Wait for the file to come up
file="/tmp/.X11-unix/X0"
for i in $(seq 1 10); do
    if [ -e "$file" ]; then
	break
    fi

    echo "Waiting for $file to be created (try $i/10)"
    sleep "$i"
done
if ! [ -e "$file" ]; then
    echo "Timing out: $file was not created"
    exit 1
fi

exec "$@"
