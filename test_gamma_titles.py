import requests

url = "https://gamma-api.polymarket.com/events"
params = {
    "limit": 1000,
    "active": "false",
    "start_date_min": "2026-03-06",
    "start_date_max": "2026-03-08T23:59:59Z",
}

r = requests.get(url, params=params)
if r.status_code == 200:
    events = r.json()
    with open("titles.txt", "w", encoding="utf-8") as f:
        f.write(f"Total events: {len(events)}\n")
        
        # See if ANY of them mention NBA, Basketball, etc
        for e in events:
            title = str(e.get('title', '')).lower()
            slug = str(e.get('slug', '')).lower()
            tags = str(e.get('tags', [])).lower()
            if 'nba' in title or 'nba' in slug or 'nba' in tags or 'basketball' in tags:
                f.write(f"FOUND NBA: {e.get('title')} | Date: {e.get('startDate')}\n")
                
        f.write("\n--- All Titles ---\n")
        for e in events[:100]:
            f.write(f"{e.get('title')} | Date: {e.get('startDate')}\n")

print("Done")
