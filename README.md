# PlayPokemonRed

![Pokemon Game Img](screenshots/POKEMON RED-2024.02.06-12.21.31.png)

## Prerequisites

- Setup a Python 3.10 virtual environment
  ```bash
  conda create -n myenv python=3.10
  conda activate myenv
  ```

- Install depencies
  ```bash
  pip install -r requirements.txt
  ```

- Ray must be installed. If not already installed, you can do so using pip:
  ```bash
  pip install ray
  ```

## Setting Up the Ray Cluster

1. Start the Ray cluster:
   ```bash
   ray up ray-cluster-config.yaml --no-config-cache --log-color true -v -y
   ```

2. Forward port 10001 for submitting jobs:
   ```bash
   ray attach ray-cluster-config.yaml -p 10001
   ```

3. Forward the dashboard for the cluster:
   ```bash
   ray dashboard /home/joe/Projects/PlayPokemonRed/ray-cluster-config.yaml
   ```

## Monitoring System Logs

You can monitor the system logs using the following commands:

- Monitor output logs:
  ```bash
  ray exec /home/joe/Projects/PlayPokemonRed/ray-cluster-config.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor.out'
  ```

- Monitor error logs:
  ```bash
  ray exec /home/joe/Projects/PlayPokemonRed/ray-cluster-config.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor.err'
  ```

- Monitor general logs:
  ```bash
  ray exec /home/joe/Projects/PlayPokemonRed/ray-cluster-config.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor.log'
  ```

## Teardown Cluster

When you're done, you can tear down the cluster using the following command:

```bash
ray down ray-cluster-config.yaml -y
```

## Start Training
*Cluster must be running*
*Please note that on line 60 of train_ray.py the S3 bucket used is hard coded please replace with one you have access to.*
- Run Command
  ```bash
  python train_ray.py
  ```

## Monitoring

- Monitoring of resources, and logs can be done via the Ray dashboard avaible [Here](http://127.0.0.1:8265) if you used the setup above.
- To monitor checkpoints, track what part of the game the AI is exploring and provide organized access to metadata. You can run the custom webapp with the below commands.
  ```bash
    cd src/monitor_app
    pip install -r requirements.txt
    python app.py
  ```
  The app will be deployed [here](http://127.0.0.1:5000/)
  *Please note that on line 27 of app.py the S3 bucket used is hard coded please replace with one you have access to.*