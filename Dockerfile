# Use official Python base image
FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /usr/src

# Install dependencies
# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the project into the container
# copy only necessary source
COPY algo/ ./algo
COPY app/ ./app
COPY utils.py ./
COPY LLM/*.py ./LLM/
COPY constants.py constants.py
COPY config.py config.py
COPY utils.py utils.py


# Expose Flask's default port
EXPOSE 5000

# Set entry point to run the Flask app
CMD ["python", "app/app.py"]


