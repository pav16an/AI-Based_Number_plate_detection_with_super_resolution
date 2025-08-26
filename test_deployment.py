#!/usr/bin/env python3
"""
Test script to verify deployment is working correctly
"""

import requests
import json
import time
import sys

def test_endpoint(url, description):
    """Test a single endpoint"""
    try:
        print(f"Testing {description}...")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"✅ {description} - OK")
            return True
        else:
            print(f"❌ {description} - Failed (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {description} - Error: {e}")
        return False

def test_api_endpoint(url, description):
    """Test API endpoint and check JSON response"""
    try:
        print(f"Testing {description}...")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {description} - OK (Records: {len(data) if isinstance(data, list) else 'N/A'})")
            return True
        else:
            print(f"❌ {description} - Failed (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {description} - Error: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ {description} - JSON Error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing License Plate Detection System Deployment")
    print("=" * 50)
    
    # Base URLs to test
    base_urls = [
        "http://localhost",      # Nginx
        "http://localhost:5000"  # Direct Flask
    ]
    
    all_tests_passed = True
    
    for base_url in base_urls:
        print(f"\n🌐 Testing {base_url}")
        print("-" * 30)
        
        # Test endpoints
        endpoints = [
            ("", "Home Page"),
            ("/webcam", "Webcam Page"),
            ("/results", "Results Page"),
        ]
        
        api_endpoints = [
            ("/api/plates", "License Plates API"),
            ("/check_db", "Database Check"),
        ]
        
        # Test regular endpoints
        for endpoint, description in endpoints:
            if not test_endpoint(f"{base_url}{endpoint}", description):
                all_tests_passed = False
        
        # Test API endpoints
        for endpoint, description in api_endpoints:
            if not test_api_endpoint(f"{base_url}{endpoint}", description):
                all_tests_passed = False
        
        # Wait between base URL tests
        if base_url != base_urls[-1]:
            time.sleep(2)
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Deployment is successful.")
        print("\n📋 Available URLs:")
        print("   - Main App: http://localhost")
        print("   - Direct Flask: http://localhost:5000")
        print("   - Live Detection: http://localhost/webcam")
        print("   - Results: http://localhost/results")
        print("   - API: http://localhost/api/plates")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the deployment.")
        sys.exit(1)

if __name__ == "__main__":
    main()