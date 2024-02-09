from flask import Flask, render_template
import sqlite3
from numpy import stack
from ray.job_submission import JobSubmissionClient
from db import setup, upsert_actor, upsert_job, upsert_checkpoint, end_actors_not_in_list, end_jobs_not_in_list
from datetime import datetime

import ray
from ray.job_submission import JobSubmissionClient
import requests
import time
import logging
import threading
import schedule
import boto3
import traceback
import json

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



app = Flask(__name__)

BUCKET='pokemon-ml'

# Setup Ray
ray_dashboard_url = "http://127.0.0.1:8265"
address='ray://localhost:10001'
ray.init(
    address=address,
    dashboard_port=8265,
)
client = JobSubmissionClient(ray_dashboard_url) 
s3 = boto3.client('s3')

def run_update_db():
    schedule.every(1).minutes.do(update_db)
    schedule.every(10).minutes.do(update_db_checkpoints)
    while True:
        schedule.run_pending()
        time.sleep(100)


def update_db():
    try:
        conn = sqlite3.connect('database.db')

        logger.info("Updating Jobs in DB")
        # Update Jobs
        jobs = client.list_jobs()
        jobs = [job for job in jobs if "/tmp/ray" in job.entrypoint]
        
        for job in jobs:
            upsert_job(
                conn,
                job.job_id,
                datetime.fromtimestamp(job.start_time/1000),
                datetime.fromtimestamp(job.end_time/1000) if job.end_time != 0 else None,
                job.status
            )

        # Update Actors
        logger.info("Updating Actors in DB")
        valid_job_ids = [job.job_id for job in jobs]

        response = requests.get(f"{ray_dashboard_url}/logical/actors")
        actors = response.json()['data']['actors']

        valid_actor_ids = []
        for k in actors.keys():
            actor = actors[k]
            # print(actor['actorClass'])
            # if actor['actorClass'] == 'TrainTrainable':
                # print(actor)
            if actor['jobId'] in valid_job_ids and actor['actorClass'] == 'RayTrainWorker':
                upsert_actor(
                    conn,
                    actor['actorId'],
                    actor['jobId'],
                    actor['address']['workerId'],
                    actor['actorClass'],
                    actor['state'],
                    datetime.fromtimestamp(actor['startTime']/1000),
                    datetime.fromtimestamp(actor['endTime']/1000) if actor['endTime'] != 0 else None
                )
                valid_actor_ids.append(actor['actorId'])
        # End Jobs and Actors not reported (May not report due to cluster restart)
        end_jobs_not_in_list(conn, valid_job_ids)
        end_actors_not_in_list(conn, valid_actor_ids)
    

    except Exception as e:
        logger.error(f"Error updating DB: {e}")
        traceback.print_exc()

def update_db_checkpoints():
    logger.info("Updating Checkpoints in DB")
    
    conn = sqlite3.connect('database.db')

    response = s3.list_objects_v2(Bucket=BUCKET, Prefix='checkpoints/LearnPokemonRed/')
    for obj in response['Contents']:
        if obj['Key'].endswith('.metadata.json'):
            response = s3.get_object(Bucket=BUCKET, Key=obj['Key'])
            content = response['Body'].read().decode('utf-8')
            checkpoint_metadata = json.loads(content)
            checkpoint_metadata['s3_key'] = 'pokemon-ml/'+obj['Key'].replace('/.metadata.json', '')
            upsert_checkpoint(
                conn,
                checkpoint_metadata.get('checkpoint_reloads'),
                checkpoint_metadata.get('episode'),
                checkpoint_metadata.get('epoch_count'),
                checkpoint_metadata.get('prev_epoch_count'),
                checkpoint_metadata.get('total_epoch_count'),
                checkpoint_metadata.get('reward'),
                checkpoint_metadata.get('avg_policy_network_loss'),
                checkpoint_metadata.get('previous_checkpoint_path'),
                checkpoint_metadata.get('actor_id'),
                checkpoint_metadata.get('job_id'),
                checkpoint_metadata.get('worker_id'),
                checkpoint_metadata.get('run_name'),
                checkpoint_metadata.get('s3_key'),
                datetime.strptime(checkpoint_metadata.get('created_at'), "%Y-%m-%d %H:%M:%S") if checkpoint_metadata.get('created_at') else None
            )



def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

@app.route('/')
def jobs_list():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row

    logger.info("Rendering jobs_list.html")

    # Connect to the SQLite database
    cursor = conn.cursor()

    # Fetch all the jobs from the database
    cursor.execute('SELECT * FROM job order by start_timestamp DESC')
    jobs = cursor.fetchall()
    print(jobs)

    # Render the template and pass the data to it
    return render_template('./jobs_list.html', jobs=jobs)

@app.route('/job/<job_id>')
def job_page(job_id):
    conn = sqlite3.connect('database.db')
    conn.row_factory = dict_factory

    logger.info(f"Rendering job_page.html for job {job_id}")

    # Connect to the SQLite database
    cursor = conn.cursor()

    # Fetch the job details from the database
    cursor.execute('SELECT * FROM job WHERE job_id = ? order by start_timestamp DESC', (job_id,))
    job = cursor.fetchone()
    
    # Fetch the actors for the given job_id from the database
    cursor.execute('SELECT * FROM actor WHERE job_id = ? order by start_timestamp DESC', (job_id,))
    actors = cursor.fetchall()
    
    for i in range(len(actors)):
        expiration_time = 3600  # 1 hour
        bucket_name = BUCKET
        actor_id = actors[i]['actor_id']
        object_key = f'screenshots/{actor_id}/-1.png'

        actors[i]['latest_image'] = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration_time
        )

    # Fetch the checkpoints for the given job_id from the database
    query = """
        SELECT c.*, c2.job_id as prev_checkpoint_job_id FROM checkpoint c
        JOIN checkpoint c2 ON c2.s3_key = c.previous_checkpoint_path
        WHERE c.job_id = ?
        order by c.created_at DESC;
    """
    cursor.execute(query, (job_id,))
    checkpoints = cursor.fetchall()

    # Render the template and pass the data to it
    return render_template('./job_page.html', job=job, actors=actors, checkpoints=checkpoints)

# Run the Flask app
if __name__ == '__main__':
    # Setup DB
    conn = sqlite3.connect('database.db')
    setup(conn)
    update_db_checkpoints()
    update_db()

    # Start the update_db thread
    update_db_thread = threading.Thread(target=run_update_db, daemon=True)
    update_db_thread.start()

    app.run()
