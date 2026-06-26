import requests
from datetime import datetime

def get_live_weather(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,cloud_cover",
            "timezone": "Asia/Kolkata"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        current = data.get('current_weather', {})
        hourly = data.get('hourly', {})
        
        # Extract values with proper fallbacks
        temperature = current.get('temperature', 0)
        rainfall = current.get('precipitation', 0)
        wind_speed = current.get('windspeed', 0)
        wind_direction = current.get('winddirection', 0)
        timestamp = current.get('time', datetime.now().isoformat())
        
        # Get humidity from hourly data
        humidity = 45  # Default fallback
        if 'relative_humidity_2m' in hourly and hourly['relative_humidity_2m']:
            humidity = hourly['relative_humidity_2m'][0]
        
        # Get cloud cover from hourly data
        cloud_cover = 20  # Default fallback
        if 'cloud_cover' in hourly and hourly['cloud_cover']:
            cloud_cover = hourly['cloud_cover'][0]
        
        # Get pressure from hourly data
        pressure = 1013  # Default fallback
        if 'surface_pressure' in hourly and hourly['surface_pressure']:
            pressure = hourly['surface_pressure'][0]
        
        return {
            'temperature': temperature,
            'rainfall': rainfall,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'pressure': pressure,
            'cloud_cover': cloud_cover,
            'timestamp': timestamp,
            'source': 'Open-Meteo'
        }
    except Exception as e:
        print(f"⚠️ Weather API error: {e}")
        return {
            'temperature': 25,
            'rainfall': 0,
            'humidity': 45,
            'wind_speed': 10,
            'wind_direction': 180,
            'pressure': 1013,
            'cloud_cover': 20,
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