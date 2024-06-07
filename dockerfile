# Use Python 3.11.4 as the base image
FROM python:3.11.4

# Set the working directory to /app
WORKDIR /app
# Install dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx


# Copy only the requirements file, to leverage Docker cache
COPY requirements.txt .

# Create and activate a virtual environment
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"
RUN /app/venv/bin/pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 80 available to the world outside this container (if needed)
EXPOSE 80

# Define environment variable (if needed)
ENV NAME World

# Command to run your application
CMD ["python", "app.py"]
