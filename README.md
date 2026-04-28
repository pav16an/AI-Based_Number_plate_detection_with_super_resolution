# License Plate Detection System

Production-ready Flask application for license plate detection and OCR using `YOLO` and `EasyOCR`.

## Project Structure

```text
app.py                  WSGI entrypoint
run.py                  Flask app factory
src/                    Application code
templates/              Web UI templates
weights/best.pt         Detection model
tests/                  Automated tests
docker/                 Dockerfiles
docker-compose.yml      Local production-style run
render.yaml             Render deployment blueprint
```

## What Was Cleaned Up

- Removed legacy duplicate folders.
- Removed obsolete helper files like old Gradio and legacy DB/config modules.
- Kept one deployable Flask app path: `app.py` -> `run.py` -> `src/`.
- Added Docker and Render-friendly runtime defaults.

## Run Locally

### Option 1: Python

1. Create a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements-prod.txt
```

3. Initialize folders and database:

```powershell
python init_project.py
```

4. Start the app:

```powershell
python run.py
```

5. Open:

```text
http://localhost:5000
```

### Option 2: Docker

```powershell
docker-compose up -d --build
```

Then open `http://localhost:5000`.

## Environment Variables

Copy `.env.example` to `.env` and edit the important values:

```text
FLASK_ENV=production
SECRET_KEY=change-this
DATABASE_PATH=data/app.db
MODEL_PATH=weights/best.pt
MODEL_DEVICE=cpu
```

## Push To Your Existing GitHub Repo

The repo already has a remote named `origin`.

1. Review changes:

```powershell
git status
```

2. Stage everything:

```powershell
git add .
```

3. Commit:

```powershell
git commit -m "Clean project structure and add deployment setup"
```

4. Push to your existing repo:

```powershell
git push origin main
```

If your branch is not `main`, check it with:

```powershell
git branch --show-current
```

and push that branch instead.

## Deploy To Render

Best choice for this app: `Render Web Service` using `Docker`.

### Why Docker on Render

- This app needs system CV libraries.
- PyTorch, EasyOCR, and OpenCV are easier to ship consistently in Docker.
- The repo now includes `render.yaml` and a Docker-based startup path.

### Steps

1. Push this repo to GitHub.
2. Sign in to Render.
3. Click `New +` -> `Blueprint` or `Web Service`.
4. Connect your GitHub repository.
5. If using Blueprint, Render reads `render.yaml` automatically.
6. If creating a Web Service manually:
   - Runtime: `Docker`
   - Dockerfile path: `docker/Dockerfile`
   - Health check path: `/health`
7. Add environment variables if needed:
   - `SECRET_KEY`
   - `MODEL_DEVICE=cpu`
   - `PORT=10000`
8. Deploy.

### Important Render Notes

- Render web services must bind to `0.0.0.0`.
- Render commonly expects the app to listen on port `10000`.
- Filesystem is ephemeral by default. If you want uploaded files or SQLite DB to persist across deploys, add a persistent disk and mount it at `/app/data` or `/app/uploads`.
- For serious production usage, replace SQLite with PostgreSQL.

## Health Check

Use:

```text
/health
```

## Tests

```powershell
pytest -q
```
