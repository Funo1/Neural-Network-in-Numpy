# Neural-Network-in-Numpy
This is the neural network I built as a novice in AI. As of 29/9/26, it can trace the sine function, but my goal is to predict more complex functions.


23/9/26:
- I relearned some of the NumPy syntax, such as the difference between ‘*’ and ‘@’. Random seeding
- I determined the nature of how I want my FNN to output; I want it to have multiple test cases x = [1, 2, 3] and have it output transformed output case y = [2, 3, 4]
- Relearned basic structure of a neural network such as how pre-activation matrices correspond for the layers, and scheme of how information is passed

25/9/26:
- I still had more I needed to understand about matrix connections and how the function is implemented.
- I intentionally struggled on my own without the help of AI to try and learn the structure. I mainly learned from rewatching a 3Blue1Brown video and NumPy documentation.
- However, I did use Claude to act almost like a teacher’s assistant, where it can’t actually give you any answers but only point you in the correct direction. That’s why I had it restricted to only give me small nudges and to never write any code for the project. Every line of the code was written by me or taken from the respective library’s documentation.

26/9/26:
- Today I focused on creating the gradient descent function with the chain rule
- Specifically, the hardest thing to understand was how the cost function was implemented, especially with respect to the activation of the previous layer’s derivative. Now I understand it can be used to calculate the previous layer’s weights in backpropagation.

27/9/26:
- Yesterday’s activities ran into today with me finishing my first draft of both backward and forward propagation. 
- I did learn more about the lambda anonymous function when creating the derivative for the leaky ReLU function. I also learned a bit about how activation functions shape the output. I chose Leaky ReLU so that I didn’t run into issues with dead neurons and so that small values can still influence predictions.

28/9/26:
- Today was spent mostly reconfiguring the network and fixing the logical errors.
- One thing I learned from the project is that it’s always better to split logic up into as many small parts as possible. If you intend to reuse code, you can run into issues with nested function definitions, such as the weight initialization in the forward pass running every time I did a forward pass, which was only supposed to happen on the training initialization.
- I also ditched my while loops and opted for for loops, as they did a better job of iterating through a given number of layers and lists corresponding to each layer.
- A major part of debugging the neural network was making sure the matrices’ dimensions aligned the way I intended them to between layers in the forward pass and transposing a matrix for backpropagation.
  - Additionally, there were some logical errors where I believe at one point I was feeding the incorrect matrix too
- All of this debugging was made easier after I learned how to use Cursor’s debugger, which is similar to a tool called Python Tutor, which we use in our CS115 course here at Stevens.

29/9/26:
- I wanted to test my network on the sine function first, as it is simple and it could actually prove my network was learning.
- Today was fixing an issue with how values are initialized and scaled before going to the neural network. For example, I scaled my inputs (z-scaling) so that they’re restricted between -1 and 1, as I kept encountering integer overflow during training.
- After scaling, the result was actually plotted, but it looked like a straight line.
- It deduced it was an error with the cost function, as it only got to ~40 epochs
  - Side note: I also increased the layers of the network to 6, and it actually got the shape of a sine curve all the way up to the first hump; however, that doesn’t fix the cost issue itself; it just makes more calculations before the same error inevitably happens. Though, it did confirm that my backprop is actually working to some extent and weights are being adjusted to become more optimal, which was a good sign (sine).
  - Additionally, the network wasn’t capturing that sine was a repeating wave since it only had data between 0 and 2pi, which later, when I tried extrapolating after finally getting the curve to match, made it streak off into nowhere, presumably because of ReLU’s slope of 1 in the activation.
  - I actually don't remember exactly what I changed to fix the network's learning deficit, but I believe it was updating the function to be an average across the size of the inputs.
    - Additionally, I updated the learning rate and loss threshold to be smaller.
