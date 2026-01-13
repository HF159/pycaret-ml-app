# File: run_with_ngrok.py
# Script to run the Streamlit app and expose it via Ngrok

import os
import subprocess
import threading
import time
from pyngrok import ngrok

def run_streamlit():
    """Run the Streamlit app"""
    print("Starting Streamlit app...")
    subprocess.run(["streamlit", "run", "main.py", "--server.headless=true"])

def main():
    """Main function to run Streamlit with Ngrok"""
    print("=" * 60)
    print("PyCaret ML App - Ngrok Launcher")
    print("=" * 60)

    # Start Streamlit in a separate thread
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()

    # Wait for Streamlit to start
    print("\nWaiting for Streamlit to start...")
    time.sleep(5)

    # Get Streamlit port (default is 8501)
    streamlit_port = 8501

    try:
        # Create ngrok tunnel
        print(f"\nCreating Ngrok tunnel to port {streamlit_port}...")
        public_url = ngrok.connect(streamlit_port, bind_tls=True)

        print("\n" + "=" * 60)
        print("✅ SUCCESS! Your app is now publicly accessible!")
        print("=" * 60)
        print(f"\n🌐 Public URL: {public_url}")
        print(f"📱 Share this URL with anyone to access your ML app!")
        print("\n" + "=" * 60)
        print("\nPress Ctrl+C to stop the server and close the tunnel")
        print("=" * 60 + "\n")

        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            ngrok.disconnect(public_url)
            print("✅ Ngrok tunnel closed. Goodbye!")

    except Exception as e:
        print(f"\n❌ Error creating Ngrok tunnel: {str(e)}")
        print("\nPlease make sure:")
        print("1. You have installed pyngrok: pip install pyngrok")
        print("2. Ngrok is properly configured")
        print("3. Port 8501 is not already in use")

if __name__ == "__main__":
    main()
