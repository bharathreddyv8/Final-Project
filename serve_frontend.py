"""
Simple HTTP Server for Frontend
Serves the frontend on http://localhost:3000
"""
import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Get the frontend directory
FRONTEND_DIR = Path(__file__).parent / "frontend"
PORT = 3000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

if __name__ == "__main__":
    os.chdir(FRONTEND_DIR)
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("="*70)
        print("MEDICAL INSURANCE FRAUD DETECTION - FRONTEND SERVER")
        print("Student: Bharath Kumar")
        print("="*70)
        print(f"\n✓ Frontend server running at: http://localhost:{PORT}")
        print(f"✓ Backend API running at: http://localhost:8000")
        print("\nOpening browser...")
        print("\nPress CTRL+C to stop the server")
        print("="*70)
        
        # Open browser
        webbrowser.open(f'http://localhost:{PORT}')
        
        # Start serving
        httpd.serve_forever()
