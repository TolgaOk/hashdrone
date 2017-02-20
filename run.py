import tensorflow as tf
import numpy as np
from dimension_reduction import autoencoder as ae
from environment import env, cross_entrophy, action_converter
import parser

warehouse_list, order_list, drone_list, product_catalogue = parser.parse('busy_day.in')
warehouse_data = np.vstack([np.array(w.products, dtype=np.float32) for w in warehouse_list])
order_data = np.vstack([np.array(o.products, dtype=np.float32) for o in order_list])
random_data = np.vstack([np.random.randint(0, 10, 400).astype(np.float32) for i in xrange(240)])

data = np.concatenate([warehouse_data, order_data, random_data])
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data = (data-mean)/std

warehouse_data = np.vstack([np.array(w.position, dtype=np.float32) for w in warehouse_list])
order_data = np.vstack([np.array(o.position, dtype=np.float32) for o in order_list])
random_data = np.vstack([np.random.randint(0, 10, 2).astype(np.float32) for i in xrange(240)])

pose_data = np.concatenate([warehouse_data, order_data, random_data])
pose_mean = np.mean(pose_data, axis=0)
pose_std = np.std(pose_data, axis=0)
pose_data = (pose_data-pose_mean)/pose_std

sess = tf.InteractiveSession()

autoencoder = ae(400, 2)
autoencoder.pose_encoder(2)

sess.run(tf.global_variables_initializer())

autoencoder.train(sess, 2001, data, 0.001)
autoencoder.pose_train(sess, 2001, pose_data, 0.001)

ce = cross_entrophy(42, 10)

for i in xrange(1):
    rewards = list()
    creatures = ce.livings_canerator(20)
    for creature in creatures:
        environment = env(creature, autoencoder, sess, 500, warehouse_list, order_list, drone_list, product_catalogue, 200, mean, std, pose_mean, pose_std)
        for it in xrange(500):
            drone_index = it%30
            environment.play(drone_index)

        print environment.turn
        print [q.qsize() for q in environment.queue_list], sum([q.qsize() for q in environment.queue_list])
        rewards.append(environment.reward)
        print rewards
