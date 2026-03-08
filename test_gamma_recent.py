import requests
import json

url = "https://gamma-api.polymarket.com/events"

test_cases = [
    {"limit": 20, "active": "false", "offset": 0},
    {"limit": 20, "active": "false", "order": "endDate", "ascending": "false"},
    {"limit": 20, "active": "false", "start_date_min": "2024-01-01"},
    {"limit": 20, "active": "false", "order": "createdAt", "ascending": "false"}
]

for i, params in enumerate(test_cases):
    print(f"\n--- Test Case {i+1}: {params} ---")
    response = requests.get(url, params=params)
    if response.status_code == 200:
        events = response.json()
        if events:
            print(f"First event date: {events[0].get('startDate')} - {events[0].get('title')}")
            print(f"Last event date: {events[-1].get('startDate')} - {events[-1].get('title')}")
        else:
            print("No events")
    else:
        print(f"Error: {response.status_code}")
