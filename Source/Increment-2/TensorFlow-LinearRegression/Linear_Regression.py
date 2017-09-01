from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = numpy.asarray([
-0.2,-0.12,-0.1,-0.21,-0.28,-0.32,-0.31,-0.33,-0.2,-0.12,-0.37,-0.24,-0.27,-0.3,-0.31,-0.21,
-0.15,-0.11,-0.28,-0.16,-0.09,-0.15,-0.27,-0.35,-0.44,-0.28,-0.23,-0.4,-0.43,-0.47,-0.42,-0.44,
-0.35,-0.34,-0.16,-0.11,-0.34,-0.4,-0.26,-0.22,-0.27,-0.21,-0.28,-0.24,-0.28,-0.21,-0.1,-0.21,
-0.21,-0.36,-0.15,-0.09,-0.17,-0.28,-0.14,-0.2,-0.15,-0.03,-0.03,-0.03,0.08,0.12,0.09,0.13,
0.25,0.12,-0.04,-0.05,-0.09,-0.09,-0.18,-0.07,0.01,0.08,-0.13,-0.15,-0.2,0.04,0.07,0.03,-0.02,
0.05,0.03,0.06,-0.2,-0.1,-0.05,-0.02,-0.07,0.07,0.02,-0.09,0.01,0.15,-0.07,-0.02,-0.11,0.18,
0.07,0.17,0.27,0.33,0.13,0.3,0.15,0.12,0.19,0.33,0.41,0.28,0.44,0.43,0.23,0.24,0.32,0.46,0.35,
0.48,0.64,0.42,0.42,0.55,0.63,0.62,0.55,0.69,0.63,0.66,0.54,0.64,0.71,0.6,0.63,0.65,0.74,0.87,0.99
])

train_Y = numpy.asarray([
-0.13,-0.16,-0.19,-0.21,-0.24,-0.26,-0.27,-0.27,-0.27,-0.26,-0.26,-0.26,-0.27,-0.26,-0.24,-0.22,-0.21,
-0.18,-0.17,-0.17,-0.2,-0.23,-0.25,-0.28,-0.31,-0.34,-0.36,-0.38,-0.39,-0.41,-0.41,-0.38,-0.34,-0.32,-0.3,
-0.29,-0.27,-0.27,-0.27,-0.27,-0.26,-0.25,-0.25,-0.24,-0.23,-0.22,-0.21,-0.21,-0.19,-0.19,-0.19,-0.19,-0.18,
-0.17,-0.16,-0.14,-0.12,-0.08,-0.03,0.01,0.05,0.08,0.11,0.11,0.09,0.06,0.02,-0.02,-0.06,-0.08,-0.07,-0.07,
-0.07,-0.08,-0.07,-0.06,-0.05,-0.04,-0.01,0.01,0.03,0.02,-0.01,-0.02,-0.03,-0.04,-0.05,-0.04,-0.03,-0.01,0,
0,0,-0.01,0,0.02,0.04,0.07,0.12,0.17,0.2,0.21,0.21,0.21,0.21,0.22,0.25,0.28,0.32,0.34,0.34,0.34,0.34,0.34,
0.35,0.38,0.41,0.44,0.46,0.49,0.52,0.54,0.56,0.59,0.61,0.62,0.62,0.63,0.63,0.63,0.63,0.65,0.68,0.74,0.79,0.85,0.91
])
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([
    -0.2,-0.1,-0.05,-0.02,-0.07,0.07,0.02,-0.09,0.01,0.15,-0.07,-0.02,-0.11,0.18,0.07,0.17,0.27,
    0.33,0.13,0.3,0.15,0.12,0.19,0.33,0.41,0.28,0.44,0.43,0.23,0.24,0.32,0.46,0.35,0.48,0.64,
    0.42,0.42,0.55,0.63,0.62,0.55,0.69,0.63,0.66,0.54,0.64,0.71,0.6,0.63,0.65,0.74,0.87,0.99
    ])
    test_Y = numpy.asarray([
    -0.03,-0.04,-0.05,-0.04,-0.03,-0.01,0,0,0,-0.01,0,0.02,0.04,0.07,0.12,0.17,0.2,0.21,0.21,
    0.21,0.21,0.22,0.25,0.28,0.32,0.34,0.34,0.34,0.34,0.34,0.35,0.38,0.41,0.44,0.46,0.49,0.52,
    0.54,0.56,0.59,0.61,0.62,0.62,0.63,0.63,0.63,0.63,0.65,0.68,0.74,0.79,0.85,0.91
    ])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
plt.show()