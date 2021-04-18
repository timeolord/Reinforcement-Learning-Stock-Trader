#!/bin/bash

cd ~/venvs/ReinforcementLearning/bin
source activate
ray stop
sshpass -p 123123 ssh linuxmsi@192.168.0.3 "cd Desktop/ ; ./stopRay.sh"
