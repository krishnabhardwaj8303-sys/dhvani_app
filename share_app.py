from pyngrok import ngrok
import time

public_url = ngrok.connect(8501)
print(f"\nYeh link team ko bhejo: {public_url}\n")
print("Yeh terminal khula rakhna jab tak team test kar rahi hai. Ctrl+C se band karo.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    ngrok.disconnect(public_url)
    print("Tunnel closed.")
