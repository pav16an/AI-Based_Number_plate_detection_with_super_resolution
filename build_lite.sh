#!/usr/bin/env bash
set -o errexit

pip install flask gunicorn
mkdir -p uploads json weights

python -c "
import sqlite3
conn = sqlite3.connect('licensePlatesDatabase.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS LicensePlates (id INTEGER PRIMARY KEY AUTOINCREMENT, start_time TEXT, end_time TEXT, license_plate TEXT, created_at TEXT DEFAULT (datetime('now')))''')
conn.commit()
conn.close()
"