import requests
import json

url = "https://gamma-api.polymarket.com/events"
# Try a few different parameter combinations
test_cases = [
    {"limit": 20, "active": "false", "slug": "nba"},
    {"limit": 20, "active": "false", "slug": "basketball"},
    {"limit": 20, "active": "false", "title": "NBA"},
    {"limit": 100, "active": "false"} # Just pull the latest 100 closed events and see what's in there
]

for i, params in enumerate(test_cases):
    print(f"\n--- Test Case {i+1}: {params} ---")
    response = requests.get(url, params=params)
    if response.status_code == 200:
        events = response.json()
        print(f"Returned {len(events)} events")
        for event in events[:3]: # print first 3
            print(f"- Title: {event.get('title')}")
            print(f"  Slug: {event.get('slug')}")
            if event.get('tags'):
                print(f"  Tags: {event.get('tags')}")
    else:
        print(f"Error {response.status_code}")
