import numpy as np
import unittest
import matplotlib.pyplot as plt


# Network initializations
rng = np.random.default_rng(52)

#^ Initializations
layers = 6
learning_rate = 0.005
loss_threshold = 0.0000001
weights = []
bias = []
cost_tracker = []

# def z_scaling(x):
#     return ((x)-np.mean(inputs))/np.std(inputs)

#^ Prediction arguements
# jittering = rng.standard_normal(size=(1,100)) * 0.02
# values = np.linspace(0, 2*np.pi, num=100, endpoint=True)
# inputs = values + jittering


# inputs_scaled = z_scaling(inputs)
# expected_outputs = np.sin(inputs)

#& test_prediction_values = np.linspace(0, 16*np.pi, num=400, endpoint=True)
#& prediction_inputs = z_scaling(test_prediction_values).reshape(1,400)
pred_inputs = np.arange(0, 2*np.pi, 0.1).flatten()
inputs_scaled = np.array([np.sin(pred_inputs),np.sin(pred_inputs + 0.1)])
expected_outputs = np.array([np.sin(pred_inputs + 0.2)])
x_pred_vals = np.arange(0, 8*np.pi, 0.1)


def pre_activation(weights, bias, nueron_inputs_scaled):
    W = weights
    b = bias
    x = nueron_inputs_scaled

    return W @ x + b

def leaky_ReLU(pre_activ):
    '''
        leaky ReLu activation function for nueron layers
    '''
    x = pre_activ
    return np.maximum(x*0.01,x)




def init_nuerons(type):
    '''
        Initializes the weights, and biases according to the number of layers and the structure of inputs_scaled
    '''
    w_length = len(weights) + 1
    b_length = len(bias) + 1

    if type == "weight" and w_length == 1:
        weights.append((rng.standard_normal(size=(64,2))) * 1)
        init_nuerons("weight")
    elif type == "weight" and w_length < layers - 1 and w_length > 1:
        weights.append((rng.standard_normal(size=(64,64))) * 0.18)
        init_nuerons("weight")
    elif type == "bias" and b_length < layers - 1:
        bias.append(rng.standard_normal(size=(64,1)))
        init_nuerons("bias")
    elif type == 'weight':
        weights.append((rng.standard_normal(size=(1,64)))* 0.18)
        return
    elif type == 'bias':
        bias.append(rng.standard_normal(size=(1,1)))
        return
    else:
        print("There's an error involving appending nueron components")




#! move weight initializations outside of the function called inside train so that each time they aren't 
#! intialized when the foward pass is made
def forward_pass(inputs_scaled):
    '''
        It should invoke pre-activation, and activations on
    '''
    activated_input = [inputs_scaled]

    global weights
    global bias

    for activations in range(0, layers - 1, 1):
        pre_activ = pre_activation(weights[activations], bias[activations], activated_input[activations])
        if activations < layers - 2:
            activated_input.append(leaky_ReLU(pre_activ))
        elif activations == layers - 2:
            activated_input.append(pre_activ)

    result = activated_input

    return result

def loss_calculator(outputs):
    return np.sum((outputs - expected_outputs) ** 2)/outputs.shape[1]


def backward_pass(net_result):
    '''
        Calculates the nudge for weights, bias, and the activation as part of the chain rule for the previous layer to establish how
        much each weight should be adjusted. This function changes the "weights" and "bias" respective values to more accurate weights

        The cost is returned to stop the training loop at a certain level of optimzation.
    '''
    global weights
    global bias

    new_Weights = []
    new_Baises = []
    cost_derivatives = [(2 * (net_result[-1] - expected_outputs))/net_result[-1].shape[1]]
    for activations in range(0, layers - 1, 1):

        # Nudged weight calculation
        back_roll = -(activations+1)
        prev_lay_activation = net_result[-(activations+2)]

        z_pre = weights[back_roll] @ prev_lay_activation + bias[back_roll]

        Leaky_ReLU_derv = np.piecewise(z_pre, [z_pre == 0, z_pre != 0], [0.01, lambda z_pre: np.maximum(0.01*z_pre,z_pre)/z_pre ])
        if activations == 0:
            common_derv = cost_derivatives[activations]
        elif activations > 0:
            common_derv = Leaky_ReLU_derv * cost_derivatives[activations]

        w_nudge =  (common_derv) @ prev_lay_activation.T * learning_rate
        b_nudge =  np.sum(common_derv,axis=1, keepdims=True) * learning_rate
        d_cost_activation = weights[back_roll].T @ common_derv

        new_weight = weights[back_roll] - w_nudge
        new_bias = bias[back_roll] - b_nudge

        new_Weights.insert(0, new_weight)
        new_Baises.insert(0, new_bias)

        cost_derivatives.append(d_cost_activation)

    weights = new_Weights
    bias = new_Baises






def train_network (inputs_scaled):
    '''
        Initializes the training loop with the inputed weights and updates weights/bias each time it calls backward_pass.
        The model trains the nuerons are within the tolerance of loss threshold.
    '''
    # Initializing nueron components

    global weights
    global bias

    weights.append(init_nuerons("weight"))
    bias.append(init_nuerons("bias"))
    if bias[-1] == None:
        bias.pop()
    if weights[-1] == None:
        weights.pop()

    # Forward pass, backward pass loop
    epochs = 0

    while epochs < 5000000:
        network_output = forward_pass(inputs_scaled)
        cost = loss_calculator(network_output[-1])

        if cost > loss_threshold:
            backward_pass(network_output)
            epochs += 1
        else:
            break

        if epochs % 2000 == 0:
            print(f"Current cost: {cost}")
        if epochs % 20 == 0:
            cost_tracker.append(cost)
    print(f"Final cost is: {cost}")
    print(f"Cost tracker: \n {cost_tracker}")
    print(f"Number of Epochs: {epochs}")

    print(f"Weights: \n{weights}")
    print(f"Bias: \n{bias}")


train_network(inputs_scaled)


def run_trained_model(arr):
    '''
        Extends the intial input list with forward pass predictions
    '''
    output = []
    output.append(arr[0])
    output.append(arr[1])

    for i in range(0, len(x_pred_vals)-2):
        in_pred = np.array(output[i:i+2]).reshape(2,1)
        output.append(forward_pass(in_pred)[-1].item())

    return output

prediction_starter = np.array([np.sin(0), np.sin(0.1)])

model_prediction = np.array(run_trained_model(prediction_starter))


plt.style.use('_mpl-gallery')


# make data
x = x_pred_vals
y = model_prediction.flatten()
y_o = np.sin(x_pred_vals)


# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)
ax.plot(x, y_o, linewidth=2.0)

plt.show()



# forward_pass(inputs_scaled)








# class testingNN(unittest.TestCase):

#     def test_same_y_levels_case_1(self):
#         self.assertAlmostEqual(sine_pred(np.pi/4),np.sin(np.pi/4), 2)
#     def test_same_y_levels_case_2(self):
#         self.assertAlmostEqual(sine_pred((3*(np.pi))/4), np.sin((3*(np.pi))/4), 2)
#     def test_sym(self):
#         self.assertAlmostEqual(sine_pred(np.pi/4),sine_pred((3*(np.pi))/4), 2)


# if __name__ == '__main__':
#     unittest.main()


