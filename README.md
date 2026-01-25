# Inference Server

Stateful FastAPI server wrapping vLLM for vision-language inference with session management.

## Quick Start

```bash
# First time setup
make install

# Start services (vLLM + FastAPI)
make start

# Check logs
make logs
```

## Warmup (Important!)

After `make start`, run warmup to preload the vision processor before adding traffic routing:

```bash
make warmup
```

## Traffic Routing (Port 443)

To expose on port 443 (requires iptables since 443 is privileged):

```bash
# Add redirect (after warmup)
sudo iptables -t nat -A PREROUTING -p tcp --dport 443 -j REDIRECT --to-port 8080

# Verify
sudo iptables -t nat -L PREROUTING -n --line-numbers

# Remove (if needed)
sudo iptables -t nat -F PREROUTING
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/add_action` | POST | Add action to session, get response |
| `/predict` | POST | Get prediction without storing response |
| `/compact` | POST | Summarize and compress session history |

## Commands

```bash
make start     # Start vLLM + FastAPI
make stop      # Stop all services
make restart   # Restart all
make logs      # Tail combined logs
make warmup    # Preload vision processor
make clean     # Remove everything
```
