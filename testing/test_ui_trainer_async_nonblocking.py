import os
import tempfile
import sqlite3
import time

from extensions_built_in.sd_trainer.UITrainer import UITrainer


class DummyJob:
    name = 'dummy'
    meta = {}
    raw_config = {}
    training_folder = None
    log_dir = None
    training_seed = None


def make_temp_db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE Job (id INTEGER PRIMARY KEY, stop INTEGER DEFAULT 0, status TEXT, info TEXT, step INTEGER, speed_string TEXT)''')
    cursor.execute('INSERT INTO Job(id) VALUES (1)')
    conn.commit()
    conn.close()
    return path


def test_run_async_operation_nonblocking(tmp_path):
    db_path = make_temp_db()
    os.environ['AITK_JOB_ID'] = '1'

    cfg = {'sqlite_db_path': db_path, 'training_folder': str(tmp_path), 'model': {'name_or_path': 'dummy_model'}}
    trainer = UITrainer(0, DummyJob(), cfg)

    # Create a coroutine that sleeps and then writes a file
    async def sleeper_and_write():
        import asyncio
        await asyncio.sleep(0.5)
        with open(os.path.join(tmp_path, 'done.txt'), 'w') as f:
            f.write('ok')

    start = time.time()
    trainer._run_async_operation(sleeper_and_write())
    elapsed = time.time() - start

    # Call should return quickly (non-blocking)
    assert elapsed < 0.1

    # wait for background task to finish
    for _ in range(20):
        if os.path.exists(os.path.join(tmp_path, 'done.txt')):
            break
        time.sleep(0.1)
    assert os.path.exists(os.path.join(tmp_path, 'done.txt'))
