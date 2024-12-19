
# Generate a requirements.txt file from poetry
FROM python:3.12 AS requirements-stage
WORKDIR /tmp
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /tmp/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Create the container
FROM python:3.12
WORKDIR /app

# Install build essentials for C++ compilation
# RUN apt-get update && apt-get install -y build-essential

COPY --from=requirements-stage /tmp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the entire app directory
COPY . /app
ENV PYTHONPATH=/app

# Run the python code
CMD ["python", "frontend/fasthtml/storm_fasthtml.py"]