# DEPROFILE Demo Webapp

This webapp provides an anonymous reviewer-facing demo for browsing Deprofile cases and chatting with a simulated patient profile.

## Local Development

1. Install frontend dependencies:

```bash
npm install
```

2. Install backend dependencies:

```bash
pip install -r api/requirements.txt
```

3. Start the backend:

```bash
python3 -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

4. Start the frontend:

```bash
npm run dev -- --host 127.0.0.1 --port 4173
```

## Anonymous Reviewer Deployment

### Recommended Architecture

- Frontend: GitHub Pages, Vercel, or Netlify
- Backend: Render, Railway, Fly.io, or any private server that can keep environment secrets
- Reviewer URL: the frontend URL only

### Why GitHub Pages Alone Is Not Enough

GitHub Pages can only host the static frontend. This project still needs a backend because:

- the profile/prompt assembly is served by FastAPI
- the built-in demo API key must stay on the server
- the 20-turn anonymous quota is enforced on the server

So the right setup is:

- publish the frontend statically
- deploy the FastAPI backend separately
- point the frontend to that backend with `VITE_API_BASE_URL`

### Frontend Environment Variables

```bash
VITE_API_BASE_URL=https://your-backend.example.com
VITE_PUBLIC_BASE_PATH=/your-repo-name/
```

Notes:

- `VITE_API_BASE_URL` is optional for local development
- `VITE_PUBLIC_BASE_PATH` should be `/` for a root domain
- for GitHub Pages under a repository path, use `/<repo-name>/`

### Backend Environment Variables

For a reviewer demo with a built-in anonymous key:

```bash
DEPROFILE_DEMO_API_KEY=your_server_side_demo_key
DEPROFILE_DEMO_MODEL=gemini-3-pro-preview
DEPROFILE_DEMO_BASE_URL=https://aidp.bytedance.net/api/modelhub/online/v2/crawl
DEPROFILE_DEMO_API_VERSION=2024-02-01
DEPROFILE_DEMO_API_TYPE=azure
DEPROFILE_DEMO_MAX_TURNS=20
```

The demo key never reaches the browser. If the reviewer leaves the API key field blank, the backend automatically uses the server-side demo key and enforces the turn limit per browser session.

## Production Notes

- The anonymous turn quota is tracked in backend memory and is suitable for a lightweight demo deployment.
- If you later need stronger quota enforcement across restarts or multiple backend replicas, move the session counter to Redis or a database.
- Keep the repository and website anonymous by avoiding personal names, lab-specific domains, and identifiable analytics.

## Build

```bash
npm run build
```
