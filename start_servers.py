import subprocess
import time

# Start FastAPI backend
backend_process = subprocess.Popen(["uvicorn", "main:app", "--reload"])

# Start React frontend
frontend_process = subprocess.Popen(["npm", "start"], cwd="./frontend")

time.sleep(10)  # Wait for servers to spin up
