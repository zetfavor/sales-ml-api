# --- The "Base Truck" ---
# Start with an official, lightweight Python 3.11 "slim" image.
FROM python:3.11-slim

# --- The "Kitchen Setup" ---
# Set the "current folder" inside the truck to /app.
WORKDIR /app

# --- "Install the Groceries" ---
# Copy *only* our lean API requirements file into the truck.
COPY requirements.api.txt .

# Run pip to install all the "groceries" from our list.
RUN pip install --no-cache-dir -r requirements.api.txt

# --- "Bring in the Staff & Tools" ---
# Copy all our local files (api.py, models/, src/) into the truck's /app folder.
# The .dockerignore file (next step) will make sure we don't copy junk.
COPY . .

# --- "Open for Business" ---
# Tell the world that the truck's "service window" is on port 80.
EXPOSE 80

# This is the command to run when the truck starts.
# It tells the "Manager" (uvicorn) to start the "Waiter" (api:app).
# We use "0.0.0.0" to mean "serve to everyone," not just localhost.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]