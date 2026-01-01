import requests
import time

# Configuration
SERVER_URL = "http://localhost:4123/v1/audio/speech"
OUTPUT_FILE = "test_output.wav"

def test_tts():
    print(f"Testing TTS Server at {SERVER_URL}...")
    
    # 1. Prepare the JSON Payload
    payload = {
        "input": "Hello! This is a test of the multilingual text to speech server.",
        "language": "en",
        "voice": "female",#neutral
        "speed": 1.0,
        "temperature": 0.7,
        "stream": False  # Use batch mode for simple testing
    }

    try:
        # 2. Send POST request (NOT GET)
        start_time = time.time()
        print("Sending request...")
        
        response = requests.post(SERVER_URL, json=payload)
        
        elapsed = time.time() - start_time

        # 3. Check response
        if response.status_code == 200:
            print(f"✅ Success! Generated in {elapsed:.2f} seconds.")
            
            # 4. Save to file
            with open(OUTPUT_FILE, "wb") as f:
                f.write(response.content)
            
            print(f"📁 Audio saved to: {OUTPUT_FILE}")
            print("You can play this file using 'aplay test_output.wav' (if installed) or download it to Windows.")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running on port 4123?")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_tts()