from utils import*

# Create an instance of the FastAPI class
app = FastAPI()

def load_model(path):
    model = models.load_model(path)
    # print(model.summary())
    return model

def predict_digit(model, data_point):
    print("Data Point Size: ", data_point.size)

    if data_point.size != (28, 28):
        return "Invalid Image Size. Upload a 28*28 Image"
    
    data_point = data_point.convert("L")
    data_point = np.array(data_point).reshape((1,784))
    
    probs = model.predict(data_point)
    predicted_digit = np.argmax(probs)

    return str(predicted_digit)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    model = load_model('LOCAL_MODEL')
    predicted_digit = predict_digit(model, image)
    return {"digit": predicted_digit}


if __name__ == "__main__":
    
    model_path = sys.argv[1]
    model = load_model(model_path)
    model.save("LOCAL_MODEL")

    # Run the FastAPI application
    import uvicorn
    uvicorn.run("task_1:app", host="127.0.0.1", port=8000)
    