#!/bin/bash

accelerate launch \
    --config_file hardware_config.yaml \
    litex_sft.py