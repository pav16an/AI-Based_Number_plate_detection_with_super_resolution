#!/usr/bin/env bash
set -o errexit

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask gunicorn opencv-python-headless numpy Pillow ultralytics easyocr

mkdir -p uploads json weights

python -c "
import sqlite3
conn = sqlite3.connect('licensePlatesDatabase.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS LicensePlates (id INTEGER PRIMARY KEY AUTOINCREMENT, start_time TEXT, end_time TEXT, license_plate TEXT, created_at TEXT DEFAULT (datetime('now')))''')
conn.commit()
conn.close()
print('Database initialized')
"