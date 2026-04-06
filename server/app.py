from openenv_cicd_triage.server.app import app as internal_app
import uvicorn
import os

app = internal_app  # expose app for OpenEnv


def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    workers = int(os.getenv("WORKERS", "1"))

    uvicorn.run(
        "server.app:app",  # IMPORTANT: point to THIS file
        host=host,
        port=port,
        workers=workers,
    )


if __name__ == "__main__":
    main()