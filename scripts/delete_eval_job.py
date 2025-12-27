import sqlite3, json, os, sys

ID = '90e9bb9b-115c-47d5-90bf-8b25e826551c'
# DB lives in the repo root
DB = os.path.join(os.getcwd(), 'aitk_db.db')

if not os.path.exists(DB):
    print('DB not found:', DB)
    sys.exit(1)

conn = sqlite3.connect(DB)
c = conn.cursor()
row = c.execute('SELECT * FROM EvalJob WHERE id=?', (ID,)).fetchone()
if not row:
    print('No EvalJob with id', ID)
    conn.close()
    sys.exit(0)

cols = [r[1] for r in c.execute("PRAGMA table_info('EvalJob')")]
data = dict(zip(cols, row))

os.makedirs(os.path.join('ui', 'stash'), exist_ok=True)
backup_path = os.path.join('ui', 'stash', f'deleted_evaljob_{ID}.json')
with open(backup_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, default=str)

c.execute('DELETE FROM EvalJob WHERE id=?', (ID,))
conn.commit()
print('Deleted EvalJob', ID)
print('Backup written to', backup_path)
conn.close()