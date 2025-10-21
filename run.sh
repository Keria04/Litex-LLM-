#!/bin/bash

accelerate launch \
    --config_file config.yaml \
    litex_sft.py