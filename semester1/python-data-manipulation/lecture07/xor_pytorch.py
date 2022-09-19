# %% [markdown]
# |x1    |x2   |out  |
# |------|-----|-----|
# |0     |0    |0    |
# |0     |1    |1    |
# |1     |0    |1    |
# |1     |1    |0    |
# 

# %%
import numpy as np
input_features = np.array([[0,0], [0,1], [1,0], [1,1]]) 
target_output = np.array([[0,1,1,0]])
target_output = target_output.reshape(4,1)

# %%
input_features

# %%
target_output

# %%
weigths = np.array([[0.1], [0.2]]) 
bias = 0.3
lr = 0.01

# %% [markdown]
# We're using the good old perceptron we used last time:
# 
# <img src="https://marcomilanesio.github.io/img/perceptron.png" width="200"/>

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sig):
    return sig * ( 1 - sig )

# %%
# COPY/PASTE from last notebook
for epoch in range(50000):
    inputs = input_features
    in_o = np.dot(inputs, weigths) + bias #feed-forward input 
    out_o = sigmoid(in_o) # feed-forward output
    error = out_o - target_output # back-propogation
    
    x = error.sum()
    if epoch % 1000 == 0:
        print(f'epoch {epoch}: Error: {x}')
        
    derr_dout = error # 1st deriv 
    dout_din = sigmoid_derivative(out_o) # 2nd deriv
    deriv = derr_dout * dout_din
    inputs = input_features.T # 3rd deriv
    deriv_final = np.dot(inputs,deriv) # that's the one we were looking for
    
    weigths -= lr * deriv_final # update weights for i in deriv:
    
    for i in deriv:
        bias -= lr * i # update bias
    
    

# %%
point = np.array([1,0])
res1 = np.dot(point, weigths) + bias # step1 
res2 = sigmoid(res1) # step2
print(res2)

# %%
point = np.array([0,0])
res1 = np.dot(point, weigths) + bias # step1 
res2 = sigmoid(res1) # step2
print(res2)

# %%
point = np.array([1,1])
res1 = np.dot(point, weigths) + bias # step1 
res2 = sigmoid(res1) # step2
print(res2)

# %% [markdown]
# **WHY?** `xor` is not linearly separable.
# 
# <img src="https://marcomilanesio.github.io/img/xor.png" width="200"/>
# 
# We would like to have something like:
# 
# <img src="https://marcomilanesio.github.io/img/xor1.png" width="200"/>
# 
# Which is not something a linear function can do.
# 
# **Solution** Add hidden layer.
# 
# <img src="https://marcomilanesio.github.io/img/xornn.png" width="200"/>
# 
# That is:
#   * 2 neurons in the input layer
#   * 2 neurons in the hidden layer
#   * 1 neuron in the output layer
#   
# In other words:
#   * 4 weigths for the hidden layer
#   * 2 weigths for the output layer
#   

# %%
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

num_neurons_input_layer = 2
num_neurons_hidden_layer = 2
num_neurons_output_layer = 1

hidden_weigths = np.random.uniform(size=(num_neurons_input_layer, num_neurons_hidden_layer))  # 2x2
hidden_bias = np.random.uniform(size=(1, num_neurons_hidden_layer))  # 1x2
output_weigths = np.random.uniform(size=(num_neurons_hidden_layer, num_neurons_output_layer))
output_bias = np.random.uniform(size=(1, num_neurons_output_layer))

# %%
output_bias

# %%
epochs = 50000
lr = 0.1

for _ in range(epochs):
    # FF input -> hidden
    hidden_layer_activation = np.dot(inputs, hidden_weigths)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    # FF hidden -> output
    output_layer_activation = np.dot(hidden_layer_output, output_weigths)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)
    
    # back-propagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(output_weigths.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # updates
    output_weigths += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weigths += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

print(f'Output: {predicted_output}')
print(f'Loss: {error}')


# %% [markdown]
# **Exercise** figure out the matrices shapes in all the operations in the previous cell.

# %%
weigths


