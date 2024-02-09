import logging
from math import e
import os
from matplotlib.style import available
import pyarrow
from pdb import run
import boto3

import ray
from ray.runtime_env import RuntimeEnv
from ray.train.tensorflow import TensorflowTrainer
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig, Result
from ray.tune import ExperimentAnalysis
from scipy import cluster

from model_training.training_loop import training_loop
logging.basicConfig(filename="./logs/Train_Ray.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO
                    )
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def collect_best_checkpoint(bucket, prefix, metric="total_reward", mode="max"):
    logger.info(f'Bucket: {bucket}, Prefix: {prefix}')
    checkpoints = []
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    paths = [cp['Prefix'] for cp in response.get('CommonPrefixes', [])]
    for path in paths:
        logger.info(f"Collecting checkpoints from: {path}")
        try:
            result = Result.from_path(f's3://{bucket}/{path}')
        except Exception as e:
            logger.info(f">> Exception: {e}")
            continue
        if result.best_checkpoints:
            for checkpoint in result.best_checkpoints:
                total_reward = checkpoint[1].get(metric, None)
                logger.info(f"Total Reward Metric:\t{total_reward}")
                if total_reward is not None:
                    checkpoints.append(checkpoint)
    
    if len(checkpoints) == 0:
        logger.info(f"No checkpoints found")
        return None, None
    
    best_checkpoint, best_checkpoint_metrics = max(
        checkpoints, key=lambda checkpoint: checkpoint[1].get("total_reward")
    )
    logger.info(f"Best checkpoint: {best_checkpoint} with Total Reward: {best_checkpoint_metrics['total_reward']}")
    return best_checkpoint, best_checkpoint_metrics


if __name__ == "__main__":
    logger.info("Starting Ray")
    bucket='pokemon-ml'
    run_name = 'LearnPokemonRed'
    path='checkpoints/'
    storage_path = f's3://{bucket}/{path}'
    address='ray://localhost:10001'

    run_config = RunConfig(
        # This will store checkpoints on S3.
        name=run_name,
        storage_path=storage_path,
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            # *Best* checkpoints are determined by these params:
            checkpoint_score_attribute="total_reward",
            checkpoint_score_order="max",
        ),
        failure_config=FailureConfig(
            max_failures=0,
        ),
    )

    training_loop_config = {
        'env_config': {
            'tick_freq': 60,
            'emulation_speed': 5,
            'window_type': 'SDL2', #SDL2
            'disable_input': True,
        },
        'screenshots_s3_bucket': 'pokemon-ml',
        'screenshots_s3_path': 'screenshots/',
        'screen_shot_freq': 50,
        'model_save_path': 'C:\\Users\\josep\\Projects\\PlayPokemonRed\\artifacts',
        'max_episodes': 100,
        'max_epochs': 1000,
        'checkpoint_freq': 10,
        'temperature': 0.15,
        'gb_path': 'https://pokemon-ml.s3.amazonaws.com/PokemonRed.gb',
        'init_state': 'https://pokemon-ml.s3.amazonaws.com/has_pokedex_nballs.state',
        'checkpoint_reload':0,
    }

    context = ray.init(
        address=address,
        log_to_driver=True,
        dashboard_port=8265,
        include_dashboard=True,
        runtime_env={
            'working_dir': ".",
            'pip': os.path.join("../requirements.txt"),
            "env_vars": {"PYTHONFAULTHANDLER": "1"},
        },
    )
    ray.cluster_resources()
    # x = ray.state.jobs()
    # print(x)
    print('Resources:')

    available_resources = ray.available_resources()
    cluster_resources = ray.cluster_resources()
    for key in available_resources:
        print(f"Resource: {key}, Available: {available_resources[key]}, Cluster: {cluster_resources[key]}")


    for i in range(40):

        logger.info(f"######## Training Loop: {i} Started ########")
        training_loop_config['checkpoint_reload'] = i
        
        best_checkpoint, best_checkpoint_metrics = collect_best_checkpoint(bucket, f'{path}{run_name}/', metric="total_reward", mode="max")

        if best_checkpoint:
            logger.info(f"Restoring from checkpoint: {best_checkpoint}")
        else:
            logger.info(f"No best checkpoint found")
            
        trainer = TensorflowTrainer(
            train_loop_per_worker=training_loop,
            scaling_config=ScalingConfig(
                num_workers=4,
                # use_gpu=True,
                resources_per_worker={"CPU": 1, "GPU": 0}
            ),
            train_loop_config=training_loop_config,
            run_config=run_config,
            resume_from_checkpoint=best_checkpoint,
            metadata={
                'run_name': run_name,
            }
        )

        result = trainer.fit()
        if result.error:
            assert isinstance(result.error, Exception)
            logger.info("Got exception:", result.error)

        # logger.info(result)
        # logger.info(result.checkpoint)

        result_path: str = result.path
        result_filesystem: pyarrow.fs.FileSystem = result.filesystem
        logger.info(f"Results location (fs, path) = ({result_filesystem}, {result_path})")

        logger.info(f"######## Training Loop: {i} Completed ########")
