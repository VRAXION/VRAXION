#!/bin/bash
# Run inverse tick test with nohup so it survives Bash tool timeout
cd /home/user/VRAXION
nohup python v4/research/v22/tests/inverse_tick_test.py > /home/user/VRAXION/inverse_tick_output.log 2>&1 &
echo "PID: $!"
echo "Output: /home/user/VRAXION/inverse_tick_output.log"
