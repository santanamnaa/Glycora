df = load_data()

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

logistic = data["model"]
le_sex = data["le_sex"]