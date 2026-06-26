import requests
from datetime import datetime

def get_live_weather(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,cloud_cover",
            "timezone": "Asia/Kolkata"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # Get current weather
        current = data.get('current_weather', {})
        
        # Get hourly data for humidity and cloud cover
        hourly = data.get('hourly', {})
        
        # Extract values with proper fallbacks
        temperature = current.get('temperature')
        rainfall = current.get('precipitation')
        wind_speed = current.get('windspeed')
        wind_direction = current.get('winddirection')
        timestamp = current.get('time')
        
        # Get humidity from hourly data (first value)
        humidity = None
        if 'relative_humidity_2m' in hourly and hourly['relative_humidity_2m']:
            humidity = hourly['relative_humidity_2m'][0]
        
        # Get cloud cover from hourly data (first value)
        cloud_cover = None
        if 'cloud_cover' in hourly and hourly['cloud_cover']:
            cloud_cover = hourly['cloud_cover'][0]
        
        # If no humidity from hourly, use a default
        if humidity is None:
            humidity = 45  # Default fallback
        
        if cloud_cover is None:
            cloud_cover = 20  # Default fallback
        
        return {
            'temperature': temperature if temperature is not None else 0,
            'rainfall': rainfall if rainfall is not None else 0,
            'humidity': humidity,
            'wind_speed': wind_speed if wind_speed is not None else 0,
            'wind_direction': wind_direction if wind_direction is not None else 0,
            'cloud_cover': cloud_cover,
            'pressure': 1013,  # Default pressure (Open-Meteo free tier doesn't provide pressure in current_weather)
            'timestamp': timestamp if timestamp else datetime.now().isoformat(),
            'source': 'Open-Meteo'
        }
    except Exception as e:
        print(f"⚠️ Weather API error: {e}")
        # Return fallback data so frontend doesn't break
        return {
            'temperature': 25,
            'rainfall': 0,
            'humidity': 45,
            'wind_speed': 10,
            'wind_direction': 180,
            'cloud_cover': 20,
            'pressure': 1013,
            'timestamp': datetime.now().isoformat(),
            'source': 'Fallback'
        }

INDIAN_CITIES = {
    'delhi': {'lat': 28.61, 'lon': 77.23},
    'mumbai': {'lat': 19.08, 'lon': 72.88},
    'bangalore': {'lat': 12.97, 'lon': 77.59},
    'chennai': {'lat': 13.08, 'lon': 80.27},
    'hyderabad': {'lat': 17.38, 'lon': 78.48},
    'kolkata': {'lat': 22.57, 'lon': 88.36},
    'pune': {'lat': 18.52, 'lon': 73.85},
    'ahmedabad': {'lat': 23.02, 'lon': 72.57},
    'jaipur': {'lat': 26.91, 'lon': 75.79},
    'lucknow': {'lat': 26.85, 'lon': 80.95},
    'nagpur': {'lat': 21.15, 'lon': 79.09},
    'indore': {'lat': 22.72, 'lon': 75.86},
    'bhopal': {'lat': 23.26, 'lon': 77.41},
    'patna': {'lat': 25.59, 'lon': 85.14},
    'guwahati': {'lat': 26.14, 'lon': 91.79}
}