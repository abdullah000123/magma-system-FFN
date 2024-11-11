# Load the saved model parameters
model_parameters = np.load('model_parameters.npz')

# Load weights and biases from the saved file
theta0 = model_parameters['theta0']
bias0 = model_parameters['bias0']

theta1 = model_parameters['theta1']
bias1 = model_parameters['bias1']

theta2 = model_parameters['theta2']
bias2 = model_parameters['bias2']
print('model loaded')
def test(test_loader):
    correct = 0
    total = 0

    for inputs, labels in test_loader:
        inputs = inputs.view(-1, no_input).numpy()  # Flatten input images
        labels = labels.numpy()  # Convert labels to NumPy array
        
        # Forward pass
        outputs = forward_propagation(inputs)

        # Get the predicted class (highest output value)
        predicted = np.argmax(outputs, axis=1)

        # Count the number of correct predictions
        correct += np.sum(predicted == labels)
        total += labels.size

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
test(test_loader)
