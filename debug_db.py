import sqlite3
from datetime import datetime

def check_database():
    """Check database structure and content"""
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        
        print("=== Database Structure ===")
        cursor.execute("PRAGMA table_info(LicensePlates)")
        columns = cursor.fetchall()
        for col in columns:
            print(f"Column: {col[1]}, Type: {col[2]}, Default: {col[4]}")
        
        print("\n=== Current Data Count ===")
        cursor.execute('SELECT COUNT(*) FROM LicensePlates')
        count = cursor.fetchone()[0]
        print(f"Total records: {count}")
        
        print("\n=== Sample Data ===")
        cursor.execute('SELECT * FROM LicensePlates LIMIT 5')
        records = cursor.fetchall()
        for record in records:
            print(record)
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

def add_test_data():
    """Add test data to database"""
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        
        # Check if created_at column exists
        cursor.execute("PRAGMA table_info(LicensePlates)")
        columns = [col[1] for col in cursor.fetchall()]
        
        current_time = datetime.now().isoformat()
        
        test_plates = [
            ('ABC123', current_time, current_time),
            ('XYZ789', current_time, current_time),
            ('TEST001', current_time, current_time),
            ('DEMO456', current_time, current_time),
            ('SAMPLE99', current_time, current_time)
        ]
        
        if 'created_at' in columns:
            for plate, start_time, end_time in test_plates:
                cursor.execute('''
                    INSERT INTO LicensePlates(license_plate, start_time, end_time, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (plate, start_time, end_time, current_time))
        else:
            for plate, start_time, end_time in test_plates:
                cursor.execute('''
                    INSERT INTO LicensePlates(license_plate, start_time, end_time)
                    VALUES (?, ?, ?)
                ''', (plate, start_time, end_time))
        
        conn.commit()
        conn.close()
        
        print(f"Added {len(test_plates)} test records")
        
    except Exception as e:
        print(f"Error adding test data: {e}")

def fix_database_structure():
    """Fix database structure"""
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        
        # Check current structure
        cursor.execute("PRAGMA table_info(LicensePlates)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'created_at' not in columns:
            print("Adding created_at column...")
            
            # Create new table with correct structure
            cursor.execute('''
                CREATE TABLE LicensePlates_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT,
                    end_time TEXT,
                    license_plate TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            ''')
            
            # Copy existing data
            cursor.execute('''
                INSERT INTO LicensePlates_new (id, start_time, end_time, license_plate, created_at)
                SELECT id, start_time, end_time, license_plate, 
                       COALESCE(start_time, datetime('now')) as created_at
                FROM LicensePlates
            ''')
            
            # Replace old table
            cursor.execute('DROP TABLE LicensePlates')
            cursor.execute('ALTER TABLE LicensePlates_new RENAME TO LicensePlates')
            
            print("Database structure fixed!")
        else:
            print("Database structure is correct")
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error fixing database: {e}")

if __name__ == "__main__":
    print("=== Checking Database ===")
    check_database()
    
    print("\n=== Fixing Database Structure ===")
    fix_database_structure()
    
    print("\n=== Adding Test Data ===")
    add_test_data()
    
    print("\n=== Final Check ===")
    check_database()