import requests

print('stats', requests.get('http://localhost:8000/statistics').json())
print('recent', requests.get('http://localhost:8000/recent-predictions').json())
