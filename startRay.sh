#!/bin/bash

cd ~/venvs/ReinforcementLearning/bin
source activate
ray start --head
cd /home/allanlinux/Desktop/ReinforcementLearningProjects
sshpass -p 123123 ssh linuxmsi@192.168.0.3 "cd Desktop/ ; ./startRay.sh"
