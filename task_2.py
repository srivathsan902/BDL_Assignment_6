from utils import*
# Create an instance of the FastAPI class
app = FastAPI()

def load_model(path):
    model = models.load_model(path)
    # print(model.summary())
    return model

def format_image(image):
    'Convert the given image of arbitrary size to 28*28 grayscale image'
    print(image.size)
    resized_image = image.resize((28, 28)).convert("L")
    image_array = np.array(resized_image)
    'Flatten the array to get a 1D array of length 784'
    flattened_image = image_array.flatten()

    return flattened_image

def predict_digit(model, data_point):
    if data_point.size != (28, 28):
        data_point = format_image(data_point).reshape((1,784))
    else:
        data_point = data_point.convert("L").reshape((1,784))

    '''Uncomment the following lines to visualize the image in the console'''
    # plt.imshow(data_point.reshape((28,28)), cmap=plt.get_cmap('gray'))
    # plt.axis('off')
    # plt.show()

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
    uvicorn.run("task_2:app", host="127.0.0.1", port=8000)

    '''
        Open the link http://127.0.0.1:8000/docs to access the Swagger UI
    '''
    