import sqlite3

def setup(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job (
            job_id TEXT PRIMARY KEY,
            start_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            end_timestamp DATETIME DEFAULT NULL,
            last_status_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS actor (
            actor_id TEXT PRIMARY KEY,
            job_id TEXT,
            worker_id TEXT,
            actor_class TEXT,
            status TEXT,
            start_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            end_timestamp DATETIME DEFAULT NULL,
            last_status_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS checkpoint (
            s3_key TEXT PRIMARY KEY,
            checkpoint_reloads INTEGER,
            episode INTEGER,
            epoch_count INTEGER,
            prev_epoch_count INTEGER,
            total_epoch_count INTEGER,
            reward REAL,
            avg_policy_network_loss REAL,
            previous_checkpoint_path TEXT,
            actor_id TEXT,
            job_id TEXT,
            worker_id TEXT,
            run_name TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_status_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')


    conn.commit()


def upsert_job(conn, job_id, start_timestamp=None, end_timestamp=None, status=None):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO job (job_id, start_timestamp, end_timestamp, status)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
        start_timestamp = COALESCE(?, start_timestamp),
        end_timestamp = COALESCE(?, end_timestamp),
        last_status_timestamp = CURRENT_TIMESTAMP,
        status = COALESCE(?, status)
    ''', (job_id, start_timestamp, end_timestamp, status, start_timestamp, end_timestamp, status))

    conn.commit()


def upsert_actor(conn, actor_id, job_id=None, worker_id=None, actor_class=None, status=None, start_timestamp=None, end_timestamp=None):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO actor (actor_id, job_id, worker_id, actor_class, status, start_timestamp, end_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(actor_id) DO UPDATE SET
        job_id = COALESCE(?, job_id),
        worker_id = COALESCE(?, worker_id),
        actor_class = COALESCE(?, actor_class),
        status = COALESCE(?, status),
        start_timestamp = COALESCE(?, start_timestamp),
        end_timestamp = COALESCE(?, end_timestamp),
        last_status_timestamp = CURRENT_TIMESTAMP
    ''', (actor_id, job_id, worker_id, actor_class, status, start_timestamp, end_timestamp, job_id, worker_id, actor_class, status, start_timestamp, end_timestamp))

    conn.commit()

def upsert_checkpoint(conn, checkpoint_reloads, episode, epoch_count, prev_epoch_count, total_epoch_count, reward, avg_policy_network_loss, previous_checkpoint_path, actor_id, job_id, worker_id, run_name, s3_key, created_at):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO checkpoint (checkpoint_reloads, episode, epoch_count, prev_epoch_count, total_epoch_count, reward, avg_policy_network_loss, previous_checkpoint_path, actor_id, job_id, worker_id, run_name, s3_key, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(s3_key) DO UPDATE SET
        checkpoint_reloads = COALESCE(?, checkpoint_reloads),
        episode = COALESCE(?, episode),
        epoch_count = COALESCE(?, epoch_count),
        prev_epoch_count = COALESCE(?, prev_epoch_count),
        total_epoch_count = COALESCE(?, total_epoch_count),
        reward = COALESCE(?, reward),
        avg_policy_network_loss = COALESCE(?, avg_policy_network_loss),
        previous_checkpoint_path = COALESCE(?, previous_checkpoint_path),
        actor_id = COALESCE(?, actor_id),
        job_id = COALESCE(?, job_id),
        worker_id = COALESCE(?, worker_id),
        run_name = COALESCE(?, run_name),
        created_at = COALESCE(?, created_at),
        last_status_timestamp= CURRENT_TIMESTAMP
    ''', (checkpoint_reloads, episode, epoch_count, prev_epoch_count, total_epoch_count, reward, avg_policy_network_loss, previous_checkpoint_path, actor_id, job_id, worker_id, run_name, s3_key, created_at, checkpoint_reloads, episode, epoch_count, prev_epoch_count, total_epoch_count, reward, avg_policy_network_loss, previous_checkpoint_path, actor_id, job_id, worker_id, run_name, created_at))

    conn.commit()


def end_jobs_not_in_list(conn, job_ids):
    cursor = conn.cursor()
    query = f'''UPDATE job
        SET end_timestamp = CURRENT_TIMESTAMP
        AND status = 'terminated'
        WHERE job_id NOT IN ({','.join('?' for _ in job_ids)})
        AND status = 'RUNNING'
    '''
    cursor.execute(query, job_ids)

    conn.commit()

def end_actors_not_in_list(conn, actor_ids):
    cursor = conn.cursor()
    query = f'''UPDATE actor
        SET end_timestamp = CURRENT_TIMESTAMP
        AND status = 'terminated'
        WHERE actor_id NOT IN ({','.join('?' for _ in actor_ids)})
        AND status = 'ALIVE'
    '''

    cursor.execute(query, actor_ids)

    conn.commit()