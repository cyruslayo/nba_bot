import requests
import pandas as pd

def search_nba_markets(min_date="2026-03-07", max_date="2026-03-07"):
    url = "https://gamma-api.polymarket.com/events"
    market_data = []
    offset = 0
    limit = 500
    
    print(f"Searching NBA markets from {min_date} to {max_date}...")
    
    while True:
        params = {
            "limit": limit,
            "active": "false",
            "start_date_min": min_date,
            "start_date_max": f"{max_date}T23:59:59Z",
            "offset": offset
        }
        r = requests.get(url, params=params)
        if r.status_code != 200:
            print(f"API Error: {r.status_code}")
            break
            
        events = r.json()
        if not events:
            break
            
        print(f"Fetched {len(events)} events at offset {offset}...")
        
        for event in events:
            is_nba = 'nba' in str(event.get('title', '')).lower() or 'nba' in str(event.get('slug', '')).lower()
            if is_nba:
                for market in event.get('markets', []):
                    market_data.append({
                        'event_title': event.get('title'),
                        'market_id': market.get('id'),
                        'start_date': event.get('startDate')
                    })
        
        if len(events) < limit:
            break # Reached the end
        offset += limit
        
        # Safety break so we don't spam Polymarket for 10 minutes
        if offset > 10000:
            print("Reached safety limit of 10,000 events checked.")
            break
            
    df = pd.DataFrame(market_data)
    return df

df = search_nba_markets()
print(f"Total NBA Markets Found: {len(df)}")
if not df.empty:
    print(df.head())
