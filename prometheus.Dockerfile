# prometheus.Dockerfile
# This tells Render how to build the Prometheus service.

# Use the official Prometheus image as a base
FROM prom/prometheus

# Copy our Render-specific configuration file into the container
COPY prometheus.render.yml /etc/prometheus/prometheus.yml

# When the container starts, run prometheus with our config file
CMD ["--config.file=/etc/prometheus/prometheus.yml"]