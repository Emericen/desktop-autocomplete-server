# Inference Server

Stateful FastAPI server wrapping vLLM for vision-language inference with session management.

## Quick Start

```bash
# Start services (vLLM + FastAPI)
make up

# Check logs
make logs
```

## Warmup

The API automatically warms up vLLM on startup. Compose ensures vLLM is healthy before starting the API container.

To manually verify readiness:

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
| `/health` | GET | Health check (returns API and vLLM status) |
| `/predict` | POST | Add action to session, get prediction |

## Commands

```bash
make up        # Start vLLM + FastAPI
make down      # Stop all services
make restart   # Restart all
make logs      # Tail combined logs
make logs-vllm # Tail vLLM logs only
make logs-api  # Tail API logs only
make shell     # Shell into vLLM container
make build     # Rebuild API container
make warmup    # Check API readiness
make clean     # Stop and remove images
```
