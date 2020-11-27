#!/bin/bash
apt-get update && apt-get upgrade -y
apt-get install -y build-essential libssl-dev libffi-dev python3-dev python3-pip
python --version
su - ubuntu
whoami
cd /home/ubuntu/
pip3 install virtualenvwrapper --user
source ~/.local/bin/virtualenvwrapper.sh
mkvirtualenv CourseProject

git clone https://github.com/sonalsharma5990/CourseProject.git
cd CourseProject/src
pip install -r requirements.txt
