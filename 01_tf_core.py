from __future__ import print_function
import tensorflow as tf

sess = tf.Session()

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # implicit dtype
print("\n")
print("node1:", node1, "node2:", node2)
print("sess.run([node1, node2]):", sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("\n")
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
print("\n")
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print("add_and_triple:", sess.run(add_and_triple, {a: 3, b: 4.5}))
