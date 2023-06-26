# Binary-MultiClassificationNeuralNetworksFromScratchNumpyArraysOnlyPython
Neural networks for binary and multiple classification from scratch (using only numpy arrays) in Python



We will discuss how to design and construct neural networks of machine learning and artifical intelligence for binary and multi-classification from scratch (using only numpy arrays) in Python. For the following examples, we'll use the Iris dataset.

Let's describe and visualize the Iris dataset first. The Iris dataset is a small but classic dataset used in evaluating classification methodologies. Most importantly, the dataset is open source; it comprises of "setosa", "versicolor", and "virginica" types with their four measures of sepal length, sepal width, petal length, and petal width. We don't need to focus too much on the details, and all we need to know is that the three types have four metrics or columns; there are 150 rows corresponding to the 50 samples of each of the three types.

To obtain the archived, open source data we'll use the following code (also included as a ipynb file in this repository):

    import pandas
    pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

Binary classification involves discriminating or classifying data according to one output label (we'll cover multi-classification afterwards) which we'll have as a 2 dimensional numpy array; we'll use 0.0 to represent one type and 1.0 to represent one other type in the following code (e.g. "setosa" and "versicolor"):

    # Binary Classification Iris Data (Labels: setosa = 1; versicolor = 0);
    import numpy as np
    import pandas as pd
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df.tail()
    _y = df.iloc[:100, 4].values
    x = df.iloc[:_y.shape[0], :4].values
    y = np.zeros((len(_y), 1)).astype(float)
    for i in range(len(_y)):
        if _y[i].endswith('setosa'):
            y[i, 0] = 1.0

Now we have a numpy array (with shape = (100, 1)) where y is the output labels (there are 50 ones and 50 zeros in total, here I'm showing one sample of each):

    y = [
         [1.0],
         [0.0],
         ]

and x which is the input data of the four features (each sample contains four features as a numpy array, there are 50 of each type, here I'm showing one sample):
  
      x = [5.1, 3.5, 1.4, 0.2]

We can construct a neural network comprising of an input layer, a hidden layer, and an output layer shown as diagram illustrating the layer connections and the direction of forward propagation in the figure below:


![Picture4](https://github.com/OriYarden/Binary-MultiClassificationNeuralNetworksFromScratchNumpyArraysOnlyPython/assets/137197657/2ee3be25-85dc-4050-8b42-263bf8cf72d7)



Each layer contains a number of nodes that function much like the neurons in a our brain which store the histories of firings of action potentials that we can describe as cellular memory and in our computational model these numpy arrays operate on the same mechanics that biological neural networks do--at least conceptually. The input layer has four nodes corresponding to the four input features in our Iris dataset plus one additional bias node (which is literally just an input of 1.0 and its function or necessity can be thought of as something like the y-intercept in a linear function in the sense that it doesn't seem like it does a whole lot but it is required for the math to be correct). The hidden layer also has four nodes plus an additional bias node. The output layer has a single node, and this output will be either 0.0 or 1.0 which corresponds to the labels in our y variable (i.e. "setosa" and "versicolor"). Forward propagation flows from one layer to the other, from left to right in the diagram; given some inputs, the neural network will provide an output (i.e. either 0.0 or 1.0). The way in which our layers process (via an activation function, or in neurons like the propagating action potential initiated when sufficient synaptic inputs depolarize the membrane potential and activate voltage-gated sodium ion channels) and activate the nodes in the next layer depends on the layer inputs as well as their weights. We'll train the weights through feedback (described later) so that certain inputs (i.e. those corresponding to "setosa" or "versicolor") generate desired outputs (i.e. correct labels of 0.0 or 1.0).

To build the neural network it is imperative that the number of nodes in each layer are precisely set so that the matrix multiplication works correctly. We can visualize WHY the number of nodes in each of the layers have to be the way they are in the weights matrix diagram where I provided the labels for the factors that influence layer dimensions in the neural network design:

![Picture3](https://github.com/OriYarden/Binary-MultiClassificationNeuralNetworksFromScratchNumpyArraysOnlyPython/assets/137197657/e414d8fc-098e-4d8f-ad71-8bf260d102ef)

The colors match the layers shown in the previous figure. We have two weights matrices since we have a three layer network; each weights matrix can be thought of as representing the connections from one layer to the next layer, so our first weights matrix must have the dimensions that matches the input layer (i.e. the number of features in the dataset plus a bias node) for its first dimension (i.e. row-dimensions) and that matches the hidden layer (i.e. the number of features in the dataset) for its the second dimension (i.e. column-dimensions). The second weights matrix must have a first dimension (i.e. row-dimensions) for its hidden layer (i.e. number of features in the dataset plus a bias node) that matches to the first weights matrix's second dimension while the second dimension of the second weights matrix must match the output layer (i.e. the number of outputs in terms of output nodes in the output layer) which is just one in this binary classification neural network design example. Thus, we can use deductive reasoning to solve for the dimensions of the weights matrices since we know how many inputs we have, we know how many outputs we want, and we know how many layers comprising the neural network model.

In contrast, designing a neural network for multiclassification in which we use all three types in the Iris dataset:

    # Multiple Classification Iris Data (Labels: setosa = [1, 0, 0]; versicolor = [0, 1, 0]; virginica = [0, 0, 1]);
    import numpy as np
    import pandas as pd
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df.tail()
    _y = df.iloc[:150, 4].values
    x = df.iloc[:_y.shape[0], :4].values
    y = np.zeros((len(_y), 3)).astype(float)
    for i in range(len(_y)):
        if _y[i].endswith('setosa'):
            y[i, 0] = 1.0
        elif _y[i].endswith('versicolor'):
            y[i, 1] = 1.0
        elif _y[i].endswith('virginica'):
            y[i, -1] = 1.0

Requires setting up the layers a bit differently:


![Picture5](https://github.com/OriYarden/Binary-MultiClassificationNeuralNetworksFromScratchNumpyArraysOnlyPython/assets/137197657/b351b131-9968-4867-9d4b-c1f495ec61fe)



Notice that we're still using the four input features from the Iris dataset, so our input layer isn't any different compared to the input layer for binary classification. What changed is that the hidden layer now has three nodes plus an additional bias node and the output layer now has three output nodes.

This is because the multi-classification neural network's weights matrices are constructed differently, and we can see that it's due to the change in factors that influence the layer dimensions shown here in the figure:


![Picture6](https://github.com/OriYarden/Binary-MultiClassificationNeuralNetworksFromScratchNumpyArraysOnlyPython/assets/137197657/12d9bbb9-b123-498a-b067-c8e4b208e0b3)


If we didn't have the weights matrix dimensions visualized in a figure, we can work backwards from the output layer back to the input layer or I mean hidden layer since we didn't change the number of input features which means we only have to change from the second weights matrix to the second dimension of the first weights matrix (i.e. the first dimension of the first weights matrix will be the same since the input layer wasn't changed).

We know we want three output labels which corresponds to three output nodes. The number of output nodes is determined by the desired number of output labels, and which dictates the second dimension (i.e. columns) of the second weights matrix. We know that the hidden layer that is projecting to our output layer has an additional bias unit so we have to add plus one node to the desired number of output labels, which is three plus one; thus, the first dimension (i.e. rows) of the second weights matrix has to be four. The input layer's projections to the hidden layer is also dictated by the desired number of output labels, which means the second dimension (i.e. columns) of the first weights matrix has to be three.

We can build a Neural Network (NN) class in Python with an init_weights method that operates using those same factors I just described for both binary and multi-classification neural networks:

    def init_weights(self):
        if self.y.shape[-1] == 1: # 1 output node only (i.e. binary classification);
            _weights1 = np.reshape(np.random.random((self.x.shape[1] + 1)*self.x.shape[1]), [self.x.shape[1] + 1, self.x.shape[1]])
            _weights2 = np.reshape(np.random.random((self.x.shape[1] + 1)*self.y.shape[1]), [self.x.shape[1] + 1, self.y.shape[1]])
        else: # more than 1 output node (i.e. multiple classification);
            _weights1 = np.reshape(np.random.random((self.x.shape[1] + 1)*self.y.shape[1]), [self.x.shape[1] + 1, self.y.shape[1]])
            _weights2 = np.reshape(np.random.random((self.y.shape[1] + 1)*self.y.shape[1]), [self.y.shape[1] + 1, self.y.shape[1]])
        return _weights1, _weights2


After 3,000 training iterations, where each "training iteration" was one sample from the Iris dataset selected randomly, we can see a clear learning curve for both binary classification:

![download (29)](https://github.com/OriYarden/Binary-MultiClassificationNeuralNetworksFromScratchNumpyArraysOnlyPython/assets/137197657/b895bb55-a1d8-4794-8de1-0b718c9d7574)





as well as for multi-classification:

![download (31)](https://github.com/OriYarden/Binary-MultiClassificationNeuralNetworksFromScratchNumpyArraysOnlyPython/assets/137197657/fa379bd2-f93f-45bb-9637-6681fa622f61)


in terms of the error decreasing over training iterations (i.e. machine learning);


The train method is shown below:

    def train(self, x, y, iterations, new_weights=False, learning_rate=1.0):
        '''
        x: must be a numpy array where columns are the features and rows are the samples;
        y: must be a numpy array where column(s) are the output labels and rows are the samples;
        '''
        self.x, self.y = x, y
        if not self.__dict__.__contains__('weights1') or new_weights:
            self.weights1, self.weights2 = self.init_weights()

        self.errors = []
        for _ in range(iterations):
            random_sample = np.random.randint(self.y.shape[0])
            input_layer = np.append(self.x[random_sample], np.ones(1), axis=0)
            input_layer.shape += (1,)

            hidden_layer = np.append(self.activation_function(self.weights1_multiplication(self.weights1, input_layer)), np.ones(1))
            hidden_layer.shape += (1,)
            output_layer = self.activation_function(self.weights2_multiplication(self.weights2, np.reshape(hidden_layer, [1, hidden_layer.shape[0]])))

            _error = self.y[random_sample] - output_layer
            self.errors.append(_error)

            feedback_weights2 = _error*output_layer*(1.0 - output_layer)*np.reshape(np.append(hidden_layer[:-1, 0], np.ones(1)), [hidden_layer.shape[0], 1])
            hidden_layer_error = self.weights2*_error*np.reshape(hidden_layer, [hidden_layer.shape[0], 1])*np.reshape(1.0 - hidden_layer, [hidden_layer.shape[0], 1])
            feedback_weights1 = hidden_layer_error[:-1, 0]*input_layer

            self.weights1 += feedback_weights1*learning_rate
            self.weights2 += feedback_weights2*learning_rate


Given a number of iterations (i.e. for loops) to train, in this example we used 3,000 iterations for training the weights to get the results previously shown, the method carries out one forward propagation pass through the network and then one backward propagation per iteration or (for) loop. The (randomly sampled) input values are fashioned into a numpy array with the additional bias node (i.e. numpy.ones(1)), pass through the activation function after the first weights matrix mulitplication, which then passes through the activation function again after the second weights matrix multiplication along with its additional bias node, and this gives us outputs from the output layer.

From here, we calculate error as the difference between the output layer's output and the labels (e.g. [0, 0, 1]) of the random sample that we forward propagated as the input values at the start of the loop iteration. We store the errors so we can track performance in terms of the differences between our neural networks outputs and the desired outputs.

The backpropagation involves providing the weights matrices feedback; we need the feedback from the second weights matrix (remember, we're backpropagating so we calculate the feedback or error on the second weights matrix before calculating the feedback or error for the first weights matrix) to be the same row-column dimensions. Next we calculate the hidden layer's error (i.e. hidden_layer_error variable), and then finally the first weights matrix feedback or error. The backpropagation operates similar to the forward propagation in that we pass input (or in the case of backpropagation, output) values to the preceding layer, then pass those values to the next layer, etc. until we're passed through all layers of the neural network and updated the synaptic I mean numpy weights matrices. Here we demonstrated how recurrent feedback networks provide an avenue through which numpy arrays can be manipulated, updated, and effectively used to classify input features according to the real-world desired output labels. This example of supervised machine learning is a subset of artificial intelligence, and although it clearly shows a huge potential in the development of technology and equipment that could outperform humans on some tasks, keep in mind that there does in fact need to be at least some sort of underlying pattern in the data in order for this type of neural network to learn. You can try plugging in a numpy array of random values, but no matter how many iterations you train the network no learning will occur because of the lack of structure in the random dataset.


Below is the entire class and its methods (I included this as an ipynb file in the repository along with the Iris data):


    import numpy as np
    from matplotlib import pyplot as plt

    class NN:
        def init_weights(self):
            if self.y.shape[-1] == 1: # 1 output node only (i.e. binary classification);
                _weights1 = np.reshape(np.random.random((self.x.shape[1] + 1)*self.x.shape[1]), [self.x.shape[1] + 1, self.x.shape[1]])
                _weights2 = np.reshape(np.random.random((self.x.shape[1] + 1)*self.y.shape[1]), [self.x.shape[1] + 1, self.y.shape[1]])
            else: # more than 1 output node (i.e. multiple classification);
                _weights1 = np.reshape(np.random.random((self.x.shape[1] + 1)*self.y.shape[1]), [self.x.shape[1] + 1, self.y.shape[1]])
                _weights2 = np.reshape(np.random.random((self.y.shape[1] + 1)*self.y.shape[1]), [self.y.shape[1] + 1, self.y.shape[1]])
            return _weights1, _weights2

        def train(self, x, y, iterations, new_weights=False, learning_rate=1.0):
            '''
            x: must be a numpy array where columns are the features and rows are the samples;
            y: must be a numpy array where column(s) are the output labels and rows are the samples;
            '''
            self.x, self.y = x, y
            if not self.__dict__.__contains__('weights1') or new_weights:
                self.weights1, self.weights2 = self.init_weights()

            self.errors = []
            for _ in range(iterations):
                random_sample = np.random.randint(self.y.shape[0])
                input_layer = np.append(self.x[random_sample], np.ones(1), axis=0)
                input_layer.shape += (1,)

                hidden_layer = np.append(self.activation_function(self.weights1_multiplication(self.weights1, input_layer)), np.ones(1))
                hidden_layer.shape += (1,)
                output_layer = self.activation_function(self.weights2_multiplication(self.weights2, np.reshape(hidden_layer, [1, hidden_layer.shape[0]])))

                _error = self.y[random_sample] - output_layer
                self.errors.append(_error)

                feedback_weights2 = _error*output_layer*(1.0 - output_layer)*np.reshape(np.append(hidden_layer[:-1, 0], np.ones(1)), [hidden_layer.shape[0], 1])
                hidden_layer_error = self.weights2*_error*np.reshape(hidden_layer, [hidden_layer.shape[0], 1])*np.reshape(1.0 - hidden_layer, [hidden_layer.shape[0], 1])
                feedback_weights1 = hidden_layer_error[:-1, 0]*input_layer

                self.weights1 += feedback_weights1*learning_rate
                self.weights2 += feedback_weights2*learning_rate

        @staticmethod
        def activation_function(x):
            return 1.0 / (1.0 + np.exp(-x))

        @staticmethod
        def weights1_multiplication(x, y):
            _result = np.zeros((1, x.shape[1])).astype(float)
            for row in range(x.shape[0]):
                for col in range(x.shape[1]):
                    _result[0, col] += x[row, col]*y[row]
            return _result[0]

        @staticmethod
        def weights2_multiplication(x, y):
            _result = np.zeros(x.shape[1]).astype(float)
            for row in range(x.shape[0]):
                for col in range(x.shape[1]):
                    _result[col] += x[row, col]*y[0, row]
            return _result

        def plot_performance(self):
            def moving_average(y, moving_window=30):
                _y = []
                for _x in range(len(y)):
                    _y.append(np.mean(y[:_x + 1]) if _x < moving_window else np.mean(y[_x - moving_window:_x]))
                return _y

            errors = [sum(_errors) for _errors in self.errors]
            fig = plt.figure(figsize=(15, 5))
            ax = plt.subplot(1, 3, 1)
            ax.plot(np.arange(0, len(errors), 1), errors, ls='-', lw=0.5, color=[1, 0, 0])
            ax.set_ylim(min(errors) - abs(0.1*min(errors)), max(errors) + 0.1*max(errors))
            ax.set_ylabel('Error', fontsize=15, fontweight='bold')
            ax.set_xlabel('Training Iterations', fontsize=15, fontweight='bold')
            for _axis in ['x', 'y']:
                ax.tick_params(axis=_axis, which='both', bottom='on', top=False, color='gray', labelcolor='gray')
            for _axis in ['top', 'right', 'bottom', 'left']:
                ax.spines[_axis].set_visible(False)
            ax = plt.subplot(1, 3, 2)
            zoomed_error_range = [0.5*np.mean(errors) - np.var(errors)**0.5, 0.5*np.mean(errors) + np.var(errors)**0.5]
            ax.plot(np.arange(0, len(errors), 1), errors, ls='-', lw=0.5, color=[1, 0, 0])
            ax.set_ylim(zoomed_error_range[0], zoomed_error_range[1])
            ax.set_xlabel('Training Iterations', fontsize=15, fontweight='bold')
            ax.set_title('[Zoomed in]', fontsize=15, fontweight='bold')
            for _axis in ['x', 'y']:
                ax.tick_params(axis=_axis, which='both', bottom='on', top=False, color='gray', labelcolor='gray')
            for _axis in ['top', 'right', 'bottom', 'left']:
                ax.spines[_axis].set_visible(False)
            ax = plt.subplot(1, 3, 3)
            _y = moving_average(errors)
            ax.plot(np.arange(0, len(_y), 1), _y, ls='-', lw=0.5, color=[1, 0, 0])
            ax.set_ylim(zoomed_error_range[0], zoomed_error_range[1])
            ax.set_xlabel('Training Iterations', fontsize=15, fontweight='bold')
            ax.set_title('[Smoothed & Zoomed in]', fontsize=15, fontweight='bold')
            for _axis in ['x', 'y']:
                ax.tick_params(axis=_axis, which='both', bottom='on', top=False, color='gray', labelcolor='gray')
            for _axis in ['top', 'right', 'bottom', 'left']:
                ax.spines[_axis].set_visible(False)
            fig.suptitle(f'{"Binary" if self.y.shape[-1] == 1 else "Multiple"}  Classification  Performance', fontsize=20, fontweight='bold')
            plt.show()

    nn = NN()
    nn.train(x, y, iterations=3000)
    nn.plot_performance()


which also includes the method for plotting performance or error over training iterations shown in previous figures. This Neural Network (NN) Python class is robust in that it can classify any number of labels with any number of inputs, binary or multi-classification, because it can correctly initialize the weights matrices' dimensions (... and obviously do all the rest of the math correctly, too). Training on the Iris dataset for example, we can use two, three, or even just one single input feature.
