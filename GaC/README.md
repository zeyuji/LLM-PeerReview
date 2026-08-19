# GaC Server

This directory contains the GaC inference server used by the project. The runtime code is adapted from [yaoching0/GaC](https://github.com/yaoching0/GaC) at commit `bbb2567b9445504b2fa0421d440c95123ed9bf07` and is distributed under the MIT license in `../LICENSES/GaC-MIT.txt`. See the [GaC paper](https://arxiv.org/abs/2406.12585) for the method.

GaC uses a separate environment from the main project. From this directory, install its dependencies and start the four-model server:

```bash
pip install -r requirements.txt
python gac_api_server.py --config-path configs/new_7b_4_model.yaml --host 0.0.0.0 --port 8000
```

Update the model paths and GPU memory limits in `configs/new_7b_4_model.yaml` when needed. The service exposes `GET /status` and `POST /api/generate/`.

After the server is ready, return to the repository root and run:

```bash
bash ./Script/Response_Generate/GaC_7B_Response_Generate.sh
```

The generation client is implemented in `Src/baseline/gac_generate.py`.
