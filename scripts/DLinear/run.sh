#!/bin/sh

python -u runExp.py --config "scripts/DLinear/toml/ETTh1.toml"
python -u runExp.py --config "scripts/DLinear/toml/ETTh2.toml"
python -u runExp.py --config "scripts/DLinear/toml/ETTm1.toml"
python -u runExp.py --config "scripts/DLinear/toml/ETTm2.toml"
python -u runExp.py --config "scripts/DLinear/toml/WTH.toml"
python -u runExp.py --config "scripts/DLinear/toml/ELT.toml"
python -u runExp.py --config "scripts/DLinear/toml/TRF.toml"