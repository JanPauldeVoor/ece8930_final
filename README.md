# Build Simulation Environment
The simulation environment is Dockerized and assumes you are running WSL/Linux or Mac.

First, build the Docker container according to the runtime
```
./scripts/build_docker.sh [linux/mac]
```

Run the Docker container
```
./scripts/run_docker.sh [linux/mac]
```

Once you are in the environment verify installation

## CUDA environments
In the containers terminal, run
```
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```
If CUDA is available in the system, ```CUDA Available``` should be ```True```.

## Mac Environments
If Apple's MPS hardware accelerators are availble, run
```
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}');"
```
```MPS Available``` should be set to ``` True```.


# Build Palmetto Enivronment / Locally
Instructions made with the ```cuuser_luyangz_ece_8930_advanced_robo``` account.
## Conda setup
```
module add miniforge3/24.3.0-0
conda init

# Create environment
conda create -y -n so101-env python=3.12
conda activate so101-env

# Prepare environment
conda install ffmpeg=7.1.1 -c conda-forge

pip install -r requirements.txt
```

```
/home/jdevoor/.conda/envs/so101-env/lib/python3.12/site-packages/lerobot/policies/groot/groot_n1.py"
@dataclass(kw_only=True)


https://huggingface.co/google/paligemma-3b-pt-224
```