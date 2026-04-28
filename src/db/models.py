"""
Database models for License Plate Detection System
"""

from datetime import datetime
import sqlite3
import json
from typing import List, Dict, Optional
from enum import Enum


class DetectionStatus(Enum):
    """Detection status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


class DetectionRecord:
    """Detection record model"""
    
    def __init__(self, license_plate: str, confidence: float, 
                 source: str = "unknown", timestamp: Optional[str] = None,
                 image_path: Optional[str] = None, metadata: Optional[Dict] = None):
        self.id = None
        self.license_plate = license_plate
        self.confidence = confidence
        self.source = source  # 'image', 'video', 'webcam'
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.image_path = image_path
        self.metadata = metadata or {}
        self.status = DetectionStatus.SUCCESS.value
        self.created_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'license_plate': self.license_plate,
            'confidence': self.confidence,
            'source': self.source,
            'timestamp': self.timestamp,
            'image_path': self.image_path,
            'metadata': self.metadata,
            'status': self.status,
            'created_at': self.created_at
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DetectionRecord':
        """Create from dictionary"""
        record = cls(
            license_plate=data['license_plate'],
            confidence=data['confidence'],
            source=data.get('source', 'unknown'),
            timestamp=data.get('timestamp'),
            image_path=data.get('image_path'),
            metadata=data.get('metadata', {})
        )
        record.id = data.get('id')
        record.status = data.get('status', DetectionStatus.SUCCESS.value)
        record.created_at = data.get('created_at', record.created_at)
        return record


class DatabaseManager:
    """Manager for SQLite database operations"""
    
    # SQL Schemas
    TABLES = {
        'detections': '''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT DEFAULT 'unknown',
                timestamp TEXT NOT NULL,
                image_path TEXT,
                metadata TEXT,
                status TEXT DEFAULT 'success',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''',
        'sessions': '''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_agent TEXT,
                ip_address TEXT,
                created_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                metadata TEXT
            )
        ''',
        'statistics': '''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE NOT NULL,
                total_detections INTEGER DEFAULT 0,
                successful_detections INTEGER DEFAULT 0,
                failed_detections INTEGER DEFAULT 0,
                average_confidence REAL,
                unique_plates INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        '''
    }
    
    # Indexes for performance
    INDEXES = {
        'idx_license_plate': 'CREATE INDEX IF NOT EXISTS idx_license_plate ON detections(license_plate)',
        'idx_timestamp': 'CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)',
        'idx_created_at': 'CREATE INDEX IF NOT EXISTS idx_created_at ON detections(created_at)',
        'idx_status': 'CREATE INDEX IF NOT EXISTS idx_status ON detections(status)',
        'idx_session_id': 'CREATE INDEX IF NOT EXISTS idx_session_id ON sessions(session_id)'
    }
    
    def __init__(self, db_path: str):
        """Initialize database manager"""
        self.db_path = db_path
        self.connection = None
    
    def connect(self) -> bool:
        """Connect to database"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self.connection.execute('PRAGMA foreign_keys = ON')
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            self.connection.close()
    
    def initialize(self) -> bool:
        """Initialize database schema"""
        if not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Create tables
            for table_name, schema in self.TABLES.items():
                cursor.execute(schema)
            
            # Create indexes
            for index_name, index_sql in self.INDEXES.items():
                cursor.execute(index_sql)
            
            self.connection.commit()
            print("Database initialized successfully")
            return True
        except Exception as e:
            print(f"Database initialization error: {e}")
            self.connection.rollback()
            return False
        finally:
            self.disconnect()
    
    def save_detection(self, record: DetectionRecord) -> bool:
        """Save detection record to database"""
        if not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO detections 
                (license_plate, confidence, source, timestamp, image_path, metadata, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.license_plate,
                record.confidence,
                record.source,
                record.timestamp,
                record.image_path,
                json.dumps(record.metadata),
                record.status,
                record.created_at,
                datetime.utcnow().isoformat()
            ))
            
            record.id = cursor.lastrowid
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error saving detection: {e}")
            self.connection.rollback()
            return False
        finally:
            self.disconnect()
    
    def get_detection(self, detection_id: int) -> Optional[DetectionRecord]:
        """Get detection record by ID"""
        if not self.connect():
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM detections WHERE id = ?', (detection_id,))
            row = cursor.fetchone()
            
            if row:
                data = dict(row)
                data['metadata'] = json.loads(data['metadata']) if data['metadata'] else {}
                return DetectionRecord.from_dict(data)
            return None
        except Exception as e:
            print(f"Error getting detection: {e}")
            return None
        finally:
            self.disconnect()
    
    def get_all_detections(self, limit: int = 100, offset: int = 0) -> List[DetectionRecord]:
        """Get all detection records with pagination"""
        if not self.connect():
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT * FROM detections 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                data = dict(row)
                data['metadata'] = json.loads(data['metadata']) if data['metadata'] else {}
                records.append(DetectionRecord.from_dict(data))
            
            return records
        except Exception as e:
            print(f"Error getting detections: {e}")
            return []
        finally:
            self.disconnect()
    
    def search_detections(self, license_plate: str, limit: int = 50) -> List[DetectionRecord]:
        """Search detections by license plate"""
        if not self.connect():
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT * FROM detections 
                WHERE license_plate LIKE ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (f"%{license_plate}%", limit))
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                data = dict(row)
                data['metadata'] = json.loads(data['metadata']) if data['metadata'] else {}
                records.append(DetectionRecord.from_dict(data))
            
            return records
        except Exception as e:
            print(f"Error searching detections: {e}")
            return []
        finally:
            self.disconnect()

    def delete_detection(self, detection_id: int) -> bool:
        """Delete a detection record by ID."""
        if not self.connect():
            return False

        try:
            cursor = self.connection.cursor()
            cursor.execute('DELETE FROM detections WHERE id = ?', (detection_id,))
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting detection: {e}")
            self.connection.rollback()
            return False
        finally:
            self.disconnect()

    def bulk_delete_detections(self, detection_ids: List[int]) -> int:
        """Delete multiple detection records and return the count removed."""
        if not detection_ids:
            return 0

        if not self.connect():
            return 0

        try:
            cursor = self.connection.cursor()
            placeholders = ','.join(['?'] * len(detection_ids))
            cursor.execute(f'DELETE FROM detections WHERE id IN ({placeholders})', detection_ids)
            deleted_count = cursor.rowcount
            self.connection.commit()
            return deleted_count
        except Exception as e:
            print(f"Error bulk deleting detections: {e}")
            self.connection.rollback()
            return 0
        finally:
            self.disconnect()
    
    def get_statistics(self, date: str) -> Optional[Dict]:
        """Get statistics for a specific date"""
        if not self.connect():
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM statistics WHERE date = ?', (date,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return None
        finally:
            self.disconnect()
    
    def update_statistics(self) -> bool:
        """Update daily statistics"""
        if not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            today = datetime.utcnow().date()
            
            # Get statistics for today
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(confidence) as avg_confidence,
                    COUNT(DISTINCT license_plate) as unique_plates
                FROM detections 
                WHERE DATE(created_at) = ?
            ''', (today,))
            
            stats = cursor.fetchone()
            
            # Upsert statistics
            cursor.execute('''
                INSERT OR REPLACE INTO statistics 
                (date, total_detections, successful_detections, failed_detections, average_confidence, unique_plates, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                today,
                stats[0] or 0,
                stats[1] or 0,
                stats[2] or 0,
                stats[3] or 0.0,
                stats[4] or 0,
                datetime.utcnow().isoformat()
            ))
            
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error updating statistics: {e}")
            self.connection.rollback()
            return False
        finally:
            self.disconnect()
