#!/usr/bin/env python
"""Startup script with detailed error reporting."""
import os
import sys
import traceback

print("=== Starting SquashVid ===")
print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}")
print(f"Files in /app: {os.listdir('/app') if os.path.exists('/app') else 'N/A'}")

try:
    print("Attempting to import squashvid.api...")
    import squashvid.api
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# If import succeeded, run uvicorn
port = int(os.environ.get("PORT", 8000))
print(f"Starting uvicorn on port {port}...")

import uvicorn
uvicorn.run("squashvid.api:app", host="0.0.0.0", port=port)
