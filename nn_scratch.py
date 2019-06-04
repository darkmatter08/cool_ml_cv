import numpy as np

"""
This module is a from-scratch impl of a basic feedforward NN for *regression*.
Input: 2 integers
Output: float

Dataset:
Input: height and weight
Output: probability of female.

Arch:
1. Input layer
2. layer with 2 hidden units.
3. output layer

Inspired by: https://victorzhou.com/blog/intro-to-neural-networks/
"""

def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))


def d_dx_sigmoid(x):
	return sigmoid(x) * (1-sigmoid(x))


def loss(y_gt, y_pred):
	# mean square error loss
	if type(y_gt) is np.ndarray and type(y_pred) is np.ndarray:
		assert y_gt.shape == y_pred
	return np.mean(np.square(y_gt - y_pred))


def d_dx_loss(y_gt, y_pred):
	return (y_pred - y_gt)


class NN:

	def __init__(self):
		self.w1 = np.random.normal()
		self.w2 = np.random.normal()
		self.w3 = np.random.normal()
		self.w4 = np.random.normal()
		self.w5 = np.random.normal()
		self.w6 = np.random.normal()

		# random biases as well?
		self.b1 = np.random.normal()
		self.b2 = np.random.normal()
		self.b3 = np.random.normal()


	def feedforward(self, x):
		h1 = x[0] * self.w1 + x[1] * self.w2 + self.b1
		h1 = sigmoid(h1)
		h2 = x[0] * self.w3 + x[1] * self.w4 + self.b2
		h2 = sigmoid(h2)

		o = h1 * self.w5 + h2 * self.w6 + self.b3
		o = sigmoid(o)
		return o


	def train(self, train_x, train_y):
		# update weights and biases using backprop from the training data

		epochs = 1000
		lr = 0.1

		for epoch in range(epochs):
			for x, y in zip(train_x, train_y):
				# forward prop, saving intermediate values.
				h1_in = x[0] * self.w1 + x[1] * self.w2 + self.b1
				h1 = h1_out = sigmoid(h1_in)
				h2_in = x[0] * self.w3 + x[1] * self.w4 + self.b2
				h2 = sigmoid(h2_in)

				o_in = h1 * self.w5 + h2 * self.w6 + self.b3
				o = o_out = sigmoid(o_in)

				# backprop, computing each neuron's weights
				# note: each value is held in a tmp variable so that the weight values aren't updated during
				# the computation.

				# compute weights from hidden to out:			
				tmp_w5 = self.w5 + (lr * d_dx_loss(o, y) * d_dx_sigmoid(o_in) * h1)
				tmp_w6 = self.w6 + (lr * d_dx_loss(o, y) * d_dx_sigmoid(o_in) * h2)
				tmp_b3 = self.b3 + (lr * d_dx_loss(o, y) * d_dx_sigmoid(o_in) * 1)

				# compute weights for h1
				tmp_w1 = self.w1 + (lr * d_dx_loss(o, y) * d_dx_sigmoid(o_in) * self.w5 * d_dx_sigmoid(h1_in) * x[0])
				tmp_w3 = self.w3 + (lr * d_dx_loss(o, y) * d_dx_sigmoid(o_in) * self.w5 * d_dx_sigmoid(h1_in) * x[1])
				tmp_b1 = self.b1 + (lr * d_dx_loss(o, y) * d_dx_sigmoid(o_in) * self.w5 * d_dx_sigmoid(h1_in) * 1)

				# compute weights for h2
				tmp_w2 = self.w2 + (lr * d_dx_loss(o, y) * d_dx_sigmoid(o_in) * self.w6 * d_dx_sigmoid(h2_in) * x[0])
				tmp_w4 = self.w4 + (lr * d_dx_loss(o, y) * d_dx_sigmoid(o_in) * self.w6 * d_dx_sigmoid(h2_in) * x[1])
				tmp_b2 = self.b2 + (lr * d_dx_loss(o, y) * d_dx_sigmoid(o_in) * self.w6 * d_dx_sigmoid(h2_in) * 1)

				# Do the updates for all variables in parallel.
				self.w5 = tmp_w5
				self.w6 = tmp_w6
				self.b3 = tmp_b3

				self.w1 = tmp_w1
				self.w3 = tmp_w3
				self.b1 = tmp_b1

				self.w2 = tmp_w2
				self.w4 = tmp_w4
				self.b2 = tmp_b2


			if epochs % 10 == 0:
				# compute error rate on training data...
				total_loss = sum([loss(y, self.feedforward(x)) for x, y in zip(train_x, train_y)])
				print('epoch: {} mean loss: {}'.format(epoch, total_loss / len(train_x)))


# Define dataset
train_x = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
train_y = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

nn = NN()
nn.train(train_x, train_y)


# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % nn.feedforward(emily)) # 0.954 - F
print("Frank: %.3f" % nn.feedforward(frank)) # 0.056 - M