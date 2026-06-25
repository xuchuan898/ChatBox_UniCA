# ChatBox_UniCA Frontend

Vue 3 frontend for the ChatBox_UniCA RAG API.

## Quick Start

```bash
# 1. Ensure the backend is running (port 8000)
# 2. Install dependencies
npm install

# 3. Start dev server (proxies /api to localhost:8000)
npm run dev

# 4. Open http://localhost:5173 in your browser
```

## Environment

Copy `.env.example` to `.env` to customize the API base URL:

```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

When using Vite's proxy (default), leave this unset or empty.

## Build for Production

```bash
npm run build
# Output in dist/
```