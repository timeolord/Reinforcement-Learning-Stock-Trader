#!/bin/bash
cd ~/venvs/ReinforcementLearning/bin
source activate
cd ~/Desktop/ReinforcementLearningProjects
./startRay.sh
gnome-terminal -- ./startTensorboard.sh
python TradingBot.py

