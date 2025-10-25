from fastapi import FastAPI

app: FastAPI = FastAPI()


if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app:main)