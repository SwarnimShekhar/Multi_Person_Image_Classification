 Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir opencv-python numpy scikit-learn matplotlib

# Make execute.sh executable
RUN chmod +x execute.sh

# Run execute.sh when the container launches
CMD ["./execute.sh"]