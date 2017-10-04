#!/usr/bin/env bash
# for CentOS
sudo yum install -y xvfb

# for Ubuntu
sudo apt-get install xvfb -y

# Crate fake monitor
Xvfb :99 -ac -screen 0 1280x1024x24 > /dev/null &

# Set environment
echo "export DISPLAY=:99" >> ~/.profile
source ~/.profile
