# Use the official Helion image to ensure compatible Torch/Hivemind versions
FROM helionnetwork/helion:main

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install web dependencies for the API
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir fastapi uvicorn[standard]

# Copy the gateway app from the build context (the gateway folder) into /app/gateway
COPY . ./gateway

# Default command
CMD ["python3", "gateway/app.py"]

EXPOSE 8080


