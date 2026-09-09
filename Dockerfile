FROM python:3

# Create and set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only source code
COPY src/ ./src/

# Install Jupyter and other tools for VSCode support
RUN pip install jupyter ipykernel
