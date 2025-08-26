#!/usr/bin/env bash
# Build script for Render

set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads json weights

# Initialize database
python -c "
import sqlite3
conn = sqlite3.connect('licensePlatesDatabase.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS LicensePlates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT,
        end_time TEXT,
        license_plate TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    )
''')
conn.commit()
conn.close()
print('Database initialized')
"