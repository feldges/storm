FROM python:3.12-slim
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy dependency files only
COPY pyproject.toml poetry.lock ./

# Configure poetry to not create a virtual environment inside Docker
# and install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --only main

# Copy application code
COPY . /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the python code
CMD ["python", "frontend/fasthtml/storm_fasthtml.py"]