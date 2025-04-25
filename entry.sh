#!/bin/bash

# Starting custom image mybase connected to my git.
# -----------------------------------------------
# LABEL ImageId ami-0f3e0b220a52b609f
# LABEL InstanceType g4dn.xlarge
# -----------------------------------------------

# assure that shutdown in done after 5 hours.
# sudo shutdown -h +5

sudo -u ubuntu git clone git@github.com:johanneshaertel/EASE_2025_active_learning_LLM.git /home/ubuntu/EASE_2025_active_learning_LLM

# Run installation make venv.
sudo -u ubuntu bash -c "cd /home/ubuntu/EASE_2025_active_learning_LLM && make venv"

# Start screen and run the simulation. You cann connnect using screen -r s...
sudo -u ubuntu screen -dmS s1
sudo -u ubuntu screen -S s1 -p 0 -X stuff "bash /home/ubuntu/EASE_2025_active_learning_LLM/simulations.sh\n"

# sudo -u ubuntu screen -dmS s2
# sudo -u ubuntu screen -S s2 -p 0 -X stuff "bash /home/ubuntu/EASE_2025_active_learning_LLM/simulations.sh\n"

# sudo -u ubuntu screen -dmS s3
# sudo -u ubuntu screen -S s3 -p 0 -X stuff "bash /home/ubuntu/EASE_2025_active_learning_LLM/simulations.sh\n"