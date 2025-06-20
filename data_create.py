import os
import requests

save_dir = "data/known_faces"
os.makedirs(save_dir, exist_ok=True)

driver_names = [f"Driver_{i+1}" for i in range(20)]

for name in driver_names:
    try:
        response = requests.get("https://thispersondoesnotexist.com", timeout=10)
        if response.status_code == 200:
            with open(os.path.join(save_dir, f"{name}.jpg"), "wb") as f:
                f.write(response.content)
            print(f"[✓] Downloaded {name}.jpg")
        else:
            print(f"[✗] Failed to download {name}: HTTP {response.status_code}")
    except Exception as e:
        print(f"[✗] {name}: {e}")
