# syntax=docker/dockerfile:1.4
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CLOUDSDK_CORE_DISABLE_PROMPTS=1

# System dependencies and gcloud sdk
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl gnupg gcc libc6-dev bash cron vim nano procps busybox-syslogd && \
    curl -sSL https://sdk.cloud.google.com | bash && \
    # trim down the gcloud sdk (based on https://github.com/GoogleCloudPlatform/gsutil/issues/1732)
    rm -rf /root/google-cloud-sdk/platform/bundledpythonunix && \
    rm -rf $(find /root/google-cloud-sdk/ -regex ".*/__pycache__") && \
    rm -rf /root/google-cloud-sdk/.install/.backup && \
    rm -rf /root/google-cloud-sdk/bin/anthoscli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/google-cloud-sdk/bin:${PATH}"

# Set workdir to /app
WORKDIR /app

# Install uv and copy project files for dependency installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
COPY uv.lock pyproject.toml /app/

# Configure .netrc for AR authentication and install dependencies
RUN --mount=type=secret,id=svc,target=/tmp/key.json \
    mkdir -p /root && \
    gcloud auth activate-service-account --key-file=/tmp/key.json && \
    gcloud auth print-access-token > /tmp/token && \
    echo "machine europe-west6-python.pkg.dev login oauth2accesstoken password $(cat /tmp/token)" > /root/.netrc && \
    chmod 600 /root/.netrc && \
    rm /tmp/token && \
    uv sync --no-dev --locked

# Copy remaining application code and model weights
ADD /src/product_template /app/src/product_template

# Run application
CMD ["uv", "run", "/app/src/product_template/main.py"]