# Trial website exposure with Cloudflare Tunnel

This project serves the frontend and API from the FastAPI app at http://localhost:8000.

## 1) Start the app locally

From the project root:

```bash
source .venv/bin/activate
uvicorn src.api_graph_stream:app --host 0.0.0.0 --port 8000
```

The app should be reachable at:

- http://localhost:8000/

## 2) Create the Cloudflare tunnel

In the Cloudflare dashboard:

1. Open Zero Trust > Networks > Tunnels.
2. Click Create a tunnel.
3. Choose Docker and copy the generated token.
4. Save the token in a local file:

```bash
cp docker/env/.env.cloudflare.example docker/env/.env.cloudflare
```

Then edit the copied file and replace the sample token with the real one:

```bash
nano docker/env/.env.cloudflare
```

## 3) Start the tunnel

From the project root:

```bash
docker compose -f docker/docker-compose.yml up -d cloudflared
```

## 4) Configure the public hostname in Cloudflare

In the Cloudflare Tunnel screen:

1. Click Add a public hostname.
2. Set the domain or subdomain you own, for example:
   - trial.example.com
3. Choose the service type: HTTP
4. Set the URL to:
   - http://localhost:8000
5. Save it.

Your app will then be available through the Cloudflare URL.

## 5) Important notes

- The Cloudflare tunnel runs from your machine, so your laptop must stay online while the trial is active.
- If you need the app to run in the background, use a machine that stays on or a server VM.
- If you need a password-free public trial link, you can also set Cloudflare Access rules later, but that is optional for a basic trial.

## 6) Useful commands

```bash
# Start the database services

docker compose -f docker/docker-compose.yml up -d mongo qdrant

# Start the app
source .venv/bin/activate
uvicorn src.api_graph_stream:app --host 0.0.0.0 --port 8000

# Check tunnel status

docker logs -f cloudflared
```
