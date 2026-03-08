import requests
import json
from datetime import datetime

url = "https://gamma-api.polymarket.com/events"

# Test 1: Just search "nba" and get the most recent ones regardless of active status
params1 = {
    "limit": 50,
    "slug": "nba", 
    "order": "createdAt",
    "ascending": "false"
}

# Test 2: Search specific dates, active=false
params2 = {
    "limit": 50,
    "active": "false",
    "start_date_min": "2026-03-06",
    "start_date_max": "2026-03-08T23:59:59Z",
}

# Test 3: Search specific dates, active=true
params3 = {
    "limit": 50,
    "active": "true",
    "start_date_min": "2026-03-06",
    "start_date_max": "2026-03-08T23:59:59Z",
}

for i, p in enumerate([params1, params2, params3]):
    print(f"\n--- Test {i+1}: {p} ---")
    r = requests.get(url, params=p)
    if r.status_code == 200:
        events = r.json()
        print(f"Total events returned: {len(events)}")
        nba_events = [e for e in events if 'nba' in str(e.get('title', '')).lower() or 'nba' in str(e.get('slug', '')).lower()]
        print(f"Total NBA events found in this batch: {len(nba_events)}")
        for e in nba_events[:3]:
            print(f"  - [{e.get('startDate')}] (Active: {e.get('active')}) {e.get('title')}")
            for m in e.get('markets', [])[:1]:
                print(f"      Market: {m.get('question')} (ID: {m.get('id')})")
    else:
        print(f"Error: {r.status_code}")
