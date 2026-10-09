FROM ghcr.io/osgeo/gdal:ubuntu-small-3.9.3

WORKDIR /usr/src/app

# Install Python pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip git build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --break-system-packages --ignore-installed -r requirements.txt

COPY . .
RUN pip install --no-cache-dir --break-system-packages -e .

ENTRYPOINT ["python3", "run_analysis.py"]
