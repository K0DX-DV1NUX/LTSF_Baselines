#!/bin/sh

python -u runExp.py --config "scripts/iTransformer/toml/ETTh1.toml"
python -u runExp.py --config "scripts/iTransformer/toml/ETTh2.toml"
python -u runExp.py --config "scripts/iTransformer/toml/ETTm1.toml"
python -u runExp.py --config "scripts/iTransformer/toml/ETTm2.toml"
python -u runExp.py --config "scripts/iTransformer/toml/WTH.toml"
python -u runExp.py --config "scripts/iTransformer/toml/ELT.toml"
python -u runExp.py --config "scripts/iTransformer/toml/TRF.toml"