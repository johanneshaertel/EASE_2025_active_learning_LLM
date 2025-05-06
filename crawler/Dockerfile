# Use an official Python runtime as the base image
FROM python:3.9

# Install zip command
RUN apt-get update && apt-get install -y zip

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shell script to the working directory
COPY script.sh .

# Copy the Python script to the working directory
COPY main.py .
COPY config.py .

# Make the shell script executable
RUN chmod +x script.sh

# Run the shell script when the container starts
CMD ["./script.sh"]