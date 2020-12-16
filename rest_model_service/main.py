from fastapi import FastAPI

app = FastAPI()


@app.get("/api/models")
async def get_models():
    return


@app.get("/api/models/<qualified_name>/metadata")
async def get_model_metadata(qualified_name: str):
    return


@app.post("/api/models/<qualified_name>/predict")
async def post_model_prediction(qualified_name: str):
    return

if __name__ == "__main__":
