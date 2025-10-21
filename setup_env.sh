#!/bin/bash

pip install trl
pip install peft
dpkg -i litex_0.1.11-beta_amd64.deb
cd pylitex
pip install -e .
