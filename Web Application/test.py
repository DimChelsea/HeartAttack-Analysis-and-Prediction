import pickle

# Load the model from the file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Print the contents of the model
print(model)