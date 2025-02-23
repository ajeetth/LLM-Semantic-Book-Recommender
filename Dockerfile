FROM python:3.12

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src

# Set working directory to src
WORKDIR /app/src

# Expose the port for Gradio
EXPOSE 7860

# Run the Gradio app
CMD ["python", "frontend_dashboard.py"]
