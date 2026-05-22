FROM nginx:alpine

# Drop the default nginx config and bake in the Azure-aware one.
COPY frontend/nginx.azure.conf /etc/nginx/conf.d/default.conf

# Serve the built frontend directly.
COPY frontend/index.html /usr/share/nginx/html/index.html

EXPOSE 80
