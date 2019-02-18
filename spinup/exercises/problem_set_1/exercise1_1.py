import tensorflow as tf
import numpy as np

"""

Exercise 1.1: Diagonal Gaussian Likelihood

Write a function which takes in Tensorflow symbols for the means and 
log stds of a batch of diagonal Gaussian distributions, along with a 
Tensorflow placeholder for (previously-generated) samples from those 
distributions, and returns a Tensorflow symbol for computing the log 
likelihoods of those samples.

"""

def gaussian_likelihood(x, mu, log_std):
	"""
	Args:
		x: Tensor with shape [batch, dim]
		mu: Tensor with shape [batch, dim]
		log_std: Tensor with shape [batch, dim] or [dim]
	Returns:
		Tensor with shape [batch]
	"""

	small_constant = 1e-10 #to prevent divide by 0
	log_std = tf.reshape(log_std, [-1, tf.shape(x)[1]]) #not mandatory, but reshaped to make broadcasting more understandable (see meeting#2 recording)
	to_sum = -0.5 * (((x-mu)/tf.exp(log_std)+small_constant)**2 + 2*log_std + np.log(2*np.pi))
	log_likelihood = tf.reduce_sum(to_sum, axis=1)
	return log_likelihood

if __name__ == '__main__':
	"""
	Run this file to verify your solution.
	"""
	from spinup.exercises.problem_set_1_solutions import exercise1_1_soln
	from spinup.exercises.common import print_result

	sess = tf.Session()

	dim = 10
	x = tf.placeholder(tf.float32, shape=(None, dim))
	mu = tf.placeholder(tf.float32, shape=(None, dim))
	log_std = tf.placeholder(tf.float32, shape=(dim,))

	your_gaussian_likelihood = gaussian_likelihood(x, mu, log_std)
	true_gaussian_likelihood = exercise1_1_soln.gaussian_likelihood(x, mu, log_std)

	batch_size = 32
	feed_dict = {x: np.random.rand(batch_size, dim),
				 mu: np.random.rand(batch_size, dim),
				 log_std: np.random.rand(dim)}

	your_result, true_result = sess.run([your_gaussian_likelihood, true_gaussian_likelihood],
										feed_dict=feed_dict)

	correct = np.allclose(your_result, true_result)
	print_result(correct)
