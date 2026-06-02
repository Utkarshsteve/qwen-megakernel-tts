#!/usr/bin/env bash
# Start the TTS server fully detached so it survives the SSH session closing.
cd /workspace
pkill -f tts_server.py 2>/dev/null
sleep 1
nohup python -u tts_server.py >/workspace/srv.log 2>&1 </dev/null &
echo "started pid $! -> /workspace/srv.log"
