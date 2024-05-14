@echo off
	:: Installing Microsoft Visual C++ Runtime - all versions 1.0.1 if it's not already installed
	choco install vcredist-all
	:: Installing CUDA if it's not already installed
	choco install cuda
	:: Inatalling ffmpeg if it's not already installed
	choco install ffmpeg
	:: Installing Python if it's not already installed
	choco install python -y
	:: Assuming successful installation, we ensure pip is upgraded
	python -m ensurepip --upgrade
	:: Use pip to install the packages listed in 'requirements.txt'
	pip install -r requirements.txt