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