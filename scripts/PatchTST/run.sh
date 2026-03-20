#!/bin/sh

python -u runExp.py --config "scripts/PatchTST/toml/ETTh1.toml"
python -u runExp.py --config "scripts/PatchTST/toml/ETTh2.toml"
python -u runExp.py --config "scripts/PatchTST/toml/ETTm1.toml"
python -u runExp.py --config "scripts/PatchTST/toml/ETTm2.toml"
python -u runExp.py --config "scripts/PatchTST/toml/WTH.toml"
python -u runExp.py --config "scripts/PatchTST/toml/ELT.toml"
python -u runExp.py --config "scripts/PatchTST/toml/TRF.toml"