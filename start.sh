#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

/home/administrator/workspace/openedai-whisper/.venv/bin/python whisper.py --host "0.0.0.0" --model openai/whisper-tiny.en
