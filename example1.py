import tensorflow as tf
import numpy as np

x1=np.random.rand(100).astype(np.float32)
x2=np.random.rand(100).astype(np.float32)
y=x1*0.1+x2*0.3 +0.5


w1=tf.Variable(tf.random_uniform([1],-1.0,1.0))
w2=tf.Variable(tf.random_uniform([1],-1.0,1.0))
b=tf.Variable(tf.zeros([1]))

y_hat=w1*x1+w2*x2+b
m_error=tf.reduce_mean(tf.square(y-y_hat))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(m_error)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
for step in range(100000):
	sess.run(train)
	if (step %20)==0:
		print(step,sess.run(w1),sess.run(w2),sess.run(b))
