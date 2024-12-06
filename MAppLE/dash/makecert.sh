#!/bin/sh
# openssl req -x509 -newkey rsa:2048 -keyout privkey.pem -out cert.pem -days 356 -nodes

openssl genrsa -out privkey.pem 2048

# Generate certificate
openssl req -new -x509 -key privkey.pem -out cert.pem -days 365 \
  -subj "/C=US/ST=Some-State/L=City/O=Org/OU=Unit/CN=10.0.0.20"
