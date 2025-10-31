# AI Image Detector (Go)

Simple Go web app that lets you upload an image and sends it to a Hugging Face
inference endpoint to estimate whether the picture was AI generated.

## Prerequisites

- Go 1.21+ (tested on 1.22)
- Hugging Face account + API token (optional but recommended for rate limits)

## Setup

```bash
git clone <this repo>
cd workspace  # adjust if different
go run ./cmd/server
```

Before running in production, set these environment variables:

- `HF_TOKEN` ? Hugging Face access token for private or rate-limited models
- `HF_MODEL_ID` ? override the default `orionw/ai-image-detection` model
- `HF_API_URL` ? provide a fully qualified inference endpoint URL if you prefer
- `PORT` ? server port (defaults to `8080`)

You can also use a `.env` loader like `direnv` or `dotenvx` if desired.

## Usage

1. Start the server: `go run ./cmd/server`
2. Open `http://localhost:8080`
3. Upload an image (max 5 MiB) and submit
4. The page displays the top prediction, confidence, preview, and raw JSON response

## Deployment Notes

- The app is dependency-light and compiles to a single binary.
- Add HTTPS termination via a reverse proxy (Caddy, Nginx, Fly, etc.).
- Adjust `maxUploadSize` in `cmd/server/main.go` if you need bigger files.
- For higher throughput, consider pooling `http.Client` instances per detector.
