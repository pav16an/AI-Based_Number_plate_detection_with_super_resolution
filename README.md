# License Plate Detection System

Production-ready Flask application for license plate detection and OCR using `YOLOv10` and `EasyOCR`.

## Project Structure

```text
app.py                  WSGI entrypoint
run.py                  Flask app factory
src/                    Application code
templates/              Web UI templates
weights/best.pt         Previous local detection model
weights/yolov10-license-plate.pt  Default YOLOv10 plate detector
tests/                  Automated tests
docker/                 Dockerfiles
docker-compose.yml      Local production-style run
render.yaml             Render deployment blueprint
```

## Model Upgrade

The default detector is now `weights/yolov10-license-plate.pt`, a public YOLOv10 license plate model.

- Source: `Rawzy/yolov10n-license-plate-detection` on Hugging Face
- Published validation metrics: precision `0.9726`, recall `0.9148`, mAP50 `0.9673`, mAP50-95 `0.6976`
- Why it was chosen: it loads cleanly with the existing YOLOv10 compatibility shims and performed better than the previous checkpoint on one of the repo sample images during local verification

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
MODEL_PATH=weights/yolov10-license-plate.pt
MODEL_DEVICE=cpu
```

## Predict Number Plates

### Browser

1. Start the app:

```powershell
python run.py
```

2. Open `http://localhost:5000`
3. Upload an image on `/` or use `/webcam`

### API

Use the image endpoint:

```powershell
curl -X POST http://localhost:5000/api/v1/detect/image -F "file=@data/carImage2.png"
```

### Local CLI

You can now run direct prediction without opening the browser:

```powershell
.venv\Scripts\python.exe scripts\predict_plate.py data\carImage2.png --save uploads\predicted-carImage2.jpg
```

Optional flags:

- `--confidence 0.35` to tune recall vs precision
- `--fast` for faster inference with fewer fallback passes
- `--model weights\best.pt` if you want to compare against the old detector

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
