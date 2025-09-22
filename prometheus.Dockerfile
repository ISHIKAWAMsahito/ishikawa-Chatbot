# prometheus.Dockerfile
# This tells Render how to build the Prometheus service.

# Use the official Prometheus image as a base
FROM prom/prometheus

# Copy our Grafana Cloud-specific configuration file into the container
COPY prometheus.remote.yml /etc/prometheus/prometheus.yml

# When the container starts, run prometheus with our config file
CMD ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.path=/prometheus_data"]

