#!/bin/bash
#
# Generate SSL certificates for ScholarDevClaw Docker deployment
#
# Usage:
#   ./generate-ssl.sh                    # Generate self-signed cert for localhost
#   ./generate-ssl.sh example.com        # Generate self-signed cert for domain
#   ./generate-ssl.sh --letsencrypt      # Generate Let's Encrypt cert (requires certbot)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DOMAIN="localhost"
USE_LETSENCRYPT=false
SSL_DIR="./ssl"
DAYS=365

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --letsencrypt)
      USE_LETSENCRYPT=true
      shift
      ;;
    --days)
      DAYS="$2"
      shift 2
      ;;
    --dir)
      SSL_DIR="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS] [DOMAIN]"
      echo ""
      echo "Options:"
      echo "  --letsencrypt    Use Let's Encrypt (requires certbot)"
      echo "  --days DAYS      Certificate validity in days (default: 365)"
      echo "  --dir DIR        SSL directory (default: ./ssl)"
      echo "  --help           Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                          # Self-signed cert for localhost"
      echo "  $0 example.com              # Self-signed cert for domain"
      echo "  $0 --letsencrypt example.com # Let's Encrypt cert"
      exit 0
      ;;
    *)
      DOMAIN="$1"
      shift
      ;;
  esac
done

# Create SSL directory
mkdir -p "$SSL_DIR"

echo -e "${YELLOW}Generating SSL certificates for: ${DOMAIN}${NC}"

if [ "$USE_LETSENCRYPT" = true ]; then
  # Let's Encrypt certificate
  echo -e "${YELLOW}Using Let's Encrypt...${NC}"

  # Check if certbot is installed
  if ! command -v certbot &> /dev/null; then
    echo -e "${RED}Error: certbot is not installed${NC}"
    echo "Install it with:"
    echo "  Ubuntu/Debian: sudo apt-get install certbot"
    echo "  macOS: brew install certbot"
    echo "  Or visit: https://certbot.eff.org/"
    exit 1
  fi

  # Generate certificate
  echo -e "${YELLOW}Running certbot...${NC}"
  sudo certbot certonly --standalone \
    --preferred-challenges http \
    -d "$DOMAIN" \
    --non-interactive \
    --agree-tos \
    --email admin@"$DOMAIN"

  # Copy certificates
  sudo cp "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$SSL_DIR/cert.pem"
  sudo cp "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$SSL_DIR/key.pem"
  sudo chown $(whoami):$(whoami) "$SSL_DIR/cert.pem" "$SSL_DIR/key.pem"

  echo -e "${GREEN}✓ Let's Encrypt certificate generated${NC}"
  echo -e "${YELLOW}Note: Let's Encrypt certificates expire in 90 days${NC}"
  echo -e "${YELLOW}Set up auto-renewal with: sudo certbot renew --dry-run${NC}"

else
  # Self-signed certificate
  echo -e "${YELLOW}Generating self-signed certificate...${NC}"

  # Generate private key
  openssl genrsa -out "$SSL_DIR/key.pem" 2048

  # Generate certificate signing request
  openssl req -new \
    -key "$SSL_DIR/key.pem" \
    -out "$SSL_DIR/csr.pem" \
    -subj "/C=US/ST=State/L=City/O=ScholarDevClaw/CN=$DOMAIN"

  # Generate self-signed certificate
  openssl x509 -req \
    -in "$SSL_DIR/csr.pem" \
    -signkey "$SSL_DIR/key.pem" \
    -out "$SSL_DIR/cert.pem" \
    -days "$DAYS"

  # Clean up CSR
  rm "$SSL_DIR/csr.pem"

  echo -e "${GREEN}✓ Self-signed certificate generated${NC}"
  echo -e "${YELLOW}Certificate valid for $DAYS days${NC}"
fi

# Set proper permissions
chmod 600 "$SSL_DIR/key.pem"
chmod 644 "$SSL_DIR/cert.pem"

echo ""
echo -e "${GREEN}SSL certificates generated successfully!${NC}"
echo ""
echo "Files created:"
echo "  - $SSL_DIR/cert.pem (certificate)"
echo "  - $SSL_DIR/key.pem (private key)"
echo ""
echo "To use with Docker:"
echo "  docker compose -f docker/docker-compose.prod.yml up -d"
echo ""
if [ "$USE_LETSENCRYPT" = false ]; then
  echo -e "${YELLOW}Note: Self-signed certificates will show browser warnings.${NC}"
  echo -e "${YELLOW}For production, use Let's Encrypt or your own certificates.${NC}"
fi
