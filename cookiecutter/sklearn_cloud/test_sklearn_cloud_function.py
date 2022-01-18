import time
import requests
url = 'https://europe-west1-useful-lattice-337908.cloudfunctions.net/sklearn_cloud_function'
payload = {"input_data": "5, 3 , 4, 1"}

for _ in range(1000):
   r = requests.get(url, params=payload)