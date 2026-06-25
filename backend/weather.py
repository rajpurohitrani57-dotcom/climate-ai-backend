import requests
from datetime import datetime

def get_live_weather(lat, lon):
    """
    Get REAL-TIME weather data from Open-Meteo (FREE, no API key)
    Includes: temperature, rainfall, humidity, wind speed, pressure, cloud cover
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "hourly": [
                "temperature_2m",
                "precipitation",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "surface_pressure",
                "cloud_cover"
            ],
            "timezone": "Asia/Kolkata"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        current = data.get('current_weather', {})
        hourly = data.get('hourly', {})
        
        # Extract all variables
        weather_data = {
            'temperature': current.get('temperature', 0),
            'rainfall': current.get('precipitation', 0),
            'humidity': hourly.get('relative_humidity_2m', [0])[0] if 'hourly' in data else None,
            'wind_speed': current.get('windspeed', 0),
            'wind_direction': current.get('winddirection', 0),
            'pressure': hourly.get('surface_pressure', [0])[0] if 'hourly' in data else None,
            'cloud_cover': hourly.get('cloud_cover', [0])[0] if 'hourly' in data else None,
            'timestamp': current.get('time', datetime.now().isoformat()),
            'source': 'Open-Meteo'
        }
        
        return weather_data
    except Exception as e:
        print(f"⚠️ Weather API error: {e}")
        return None

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