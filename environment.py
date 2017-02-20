import numpy as np
import parser as pr
from Queue import Queue

    # rewardcuyu uyarmayi unutma

class cross_entrophy():
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mean = np.random.normal(size=(in_dim,out_dim))
        self.std = np.random.uniform(0.001, 0.0015, size=(in_dim,out_dim))
        self.history = []

    def livings_canerator(self, n_creatures):
        self.tribe = np.array([np.random.multivariate_normal(self.mean[i, :], np.diag(self.std[i, :]), n_creatures).T
                        for i in range(self.in_dim)])
        return np.reshape(self.tribe, (-1, self.out_dim, self.in_dim))

    def new_generation(self, rewards, election_ratio):
        assert len(rewards) == self.tribe.shape[0], "Reward's length is different than the tribe's length."
        survivors_index = np.argsort(rewards)[len(rewards)*election_ratio:]
        survivors = self.tribe[survivors_index,:]
        self.mean = np.mean(survivors, axis=0)
        self.std = np.std(survivors, axis=0)
        self.history.append(np.mean(rewards))


def action_converter(output_net, model, sess, warehouses, orders, drone, max_weight, prod_catalogue, mean, std, pose_mean, pose_std, chosen_drone):
    def cypher_to_product(to_be_decoded):
        products = np.squeeze(model.decode(sess, np.expand_dims(to_be_decoded, 0)))*std + mean

        return np.argmax(products), np.rint(np.max(products))

    def spatial_constraint(x, y, data):
        a = np.array([x, y], dtype=np.float32)
        x, y = (model.pose_decode(sess, np.expand_dims(a, 0))*pose_std+pose_mean)[0]
        dists = np.sqrt(np.array([(x-datum.position[0])**2+(y-datum.position[1])**2 for datum in data]).astype(np.float32))
        return np.argmin(dists)

    def weight_constraint(i_prod, n_product, drone_):
        pw = prod_catalogue[i_prod]
        available = max_weight-drone_.load
        return n_product if (available)/pw>=n_product else int(available/pw)

    discrete_actions_dict = {0:"W", 1:"L", 2:"U", 3:"D"}
    #chosen_drone = np.argmax(output_net[:30])
    chosen_action = np.argmax(output_net[:2])
    # if chosen_action == 0:
    #     wait_turn = max(int(output_net[16]), 0)
    #     return "{} {} {}".format(chosen_drone, "W", wait_turn)
    if chosen_action == 0:
        load_position = output_net[2:4]
        load_product = cypher_to_product(output_net[4:6])
        load_index = spatial_constraint(load_position[0], load_position[1], warehouses)
        constrained_product = load_product[1] if warehouses[load_index].products[load_product[0]] >= load_product[1] else warehouses[load_index].products[load_product[0]]
        constrained_product = weight_constraint(load_product[0], constrained_product, drone[chosen_drone])
        return "{} {} {} {} {}".format(chosen_drone, "L", load_index, load_product[0], int(constrained_product))
    # elif chosen_action == 2:
    #     unload_position = output_net[8:10]
    #     unload_product = cypher_to_product(output_net[10:12])
    #     unload_index = spatial_constraint(unload_position[0], unload_position[1], warehouses)
    #     constrained_product = unload_product[1] if drone[chosen_drone].products[unload_product[0]] >= unload_product[1] else drone[chosen_drone].products[unload_product[0]]
    #     return "{} {} {} {} {}".format(chosen_drone, "U", unload_index, unload_product[0], int(constrained_product))
    elif chosen_action == 1:
        delivery_position = output_net[6:8]
        delivery_product = cypher_to_product(output_net[8:10])
        delivery_index = spatial_constraint(delivery_position[0], delivery_position[1], orders)
        constrained_product = delivery_product[1] if drone[chosen_drone].products[delivery_product[0]] >= delivery_product[1] else drone[chosen_drone].products[delivery_product[0]]
        constrained_product = constrained_product if orders[delivery_index].products[delivery_product[0]] == constrained_product else 0
        return "{} {} {} {} {}".format(chosen_drone, "D", delivery_index, delivery_product[0], int(constrained_product))

    #reward parametresi --> penalti

class env(object):
    def __init__(self, agent, model, sess, max_turn, warehouse_list, order_list, drone_list, catalogue, max_weight, mean, std, pose_mean, pose_std):
        self.reward = 0.0
        self.turn = 0
        self.max_weight = max_weight
        self.agent = agent
        self.mean, self.std, self.pose_mean, self.pose_std = mean, std, pose_mean, pose_std
        self.max_turn = max_turn
        self.warehouse_list = warehouse_list
        self.order_list = order_list
        self.drone_list = drone_list
        self.catalogue = catalogue
        self.model = model
        self.sess = sess
        self.queue_list = [Queue() for i in xrange(len(drone_list))]
        self.empty_drone = [True for i in xrange(len(drone_list))]
        self.are_queues_filled = len(drone_list)

    def play(self, drone_index):
        def state():
            wr_products = np.mean(self.model.encode(self.sess, np.array([wr.products for wr in self.warehouse_list])), axis=0)
            or_products = np.mean(self.model.encode(self.sess, np.array([o.products for o in self.order_list])), axis=0)
            dr_products = np.mean(self.model.encode(self.sess, np.array([d.products for d in self.drone_list])), axis=0)

            wr_position = np.mean(self.model.pose_encode(self.sess, np.array([wr.position for wr in self.warehouse_list])), axis=0)
            or_position = np.mean(self.model.pose_encode(self.sess, np.array([o.position for o in self.order_list])), axis=0)
            dr_position = np.mean(self.model.pose_encode(self.sess, np.array([d.position for d in self.drone_list])), axis=0)

            drone_state = np.zeros((len(self.drone_list)), dtype=np.float32)
            drone_state[drone_index] = 1.0
            return np.concatenate([drone_state, wr_position, wr_products, or_position, or_products, dr_position, dr_products])


        action = np.dot(self.agent, state())
        # action /= np.max(action[:30])
        # action += np.random.normal(size=47, scale=0.9)
        #print np.max(action[:30])
        google_output = action_converter(action, self.model, self.sess, self.warehouse_list, self.order_list, self.drone_list, self.max_weight, self.catalogue, self.mean, self.std, self.pose_mean, self.pose_std, drone_index)
        i_drone = int(google_output.split()[0])

        self.queue_list[i_drone].put(google_output)
        if True:
            #self.empty_drone[i_drone] = False
            flag = sum([1 for q in self.queue_list if q.empty()])
            if not flag:
                #self.are_queues_filled -= 1
                available_drone_indexes  = [index for index, dr in enumerate(self.drone_list) if dr.is_available]
                command_list = [self.queue_list[index].get() for index in available_drone_indexes]

                # works assigned
                for command in command_list:
                    self.assign_work(command)

                turn_to_go = min([drone.until_available for drone in self.drone_list])
                drones_to_update = [drone for drone in self.drone_list if drone.until_available == turn_to_go]
                drones_to_update = sorted(drones_to_update, key=lambda dr: dr.current_command[2], reverse=True)

                # drones completed their work
                for drone in drones_to_update:
                    self.update(drone.current_command)
                    drone.is_available = True
                    drone.current_command = ""

                # drones updated
                for  drone in self.drone_list:
                    drone.until_available -= turn_to_go

                # queue management
                # for i in available_drone_indexes:
                #     if self.queue_list[i].empty():
                #         #self.are_queues_filled += 1
                #         self.empty_drone[i] = True

                self.turn += turn_to_go
            else:
                #self.are_queues_filled -= 1
                pass



    def assign_work(self, command):

        def shortest(drone, target):
            x_drone, y_drone, x_target, y_target = drone.position+target.position
            return np.ceil(np.sqrt((x_drone - x_target)**2 + (y_drone - y_target)**2))

        act_params = command.split()
        discrete_action = act_params[1]
        i_drone = int(act_params[0])

        if discrete_action == "W":
            self.drone_list[i_drone].until_available += int(act_params[2])
            self.drone_list[i_drone].is_available = False
            self.drone_list[i_drone].current_command = command
        elif discrete_action == "L" or discrete_action == "U":
            self.drone_list[i_drone].until_available += shortest(self.drone_list[i_drone], self.warehouse_list[int(act_params[2])]) + 1
            self.drone_list[i_drone].is_available = False
            self.drone_list[i_drone].current_command = command
        elif discrete_action == "D":
            self.drone_list[i_drone].until_available += shortest(self.drone_list[i_drone], self.order_list[int(act_params[2])]) + 1
            self.drone_list[i_drone].is_available = False
            self.drone_list[i_drone].current_command = command


    def update(self, action):

        act_params = action.split()
        discrete_action = act_params[1]
        i_drone = int(act_params[0])
        if discrete_action == "W":
            pass
        elif discrete_action == "L":

            self.drone_list[i_drone].position[:] = self.warehouse_list[int(act_params[2])].position[:]
            index_pro, n_pro= map(int, act_params[3:])
            self.drone_list[i_drone].products[index_pro] += n_pro
            self.drone_list[i_drone].load += self.catalogue[index_pro]*n_pro
            self.warehouse_list[int(act_params[2])].products[index_pro] -= n_pro
        elif discrete_action == "U":

            self.drone_list[i_drone].position[:] = self.warehouse_list[int(act_params[2])].position[:]
            index_pro, n_pro= map(int, act_params[3:])
            self.drone_list[i_drone].products[index_pro] -= n_pro
            self.drone_list[i_drone].load -= self.catalogue[index_pro]*n_pro
            self.warehouse_list[int(act_params[2])].products[index_pro] += n_pro
        elif discrete_action == "D":

            self.drone_list[i_drone].position[:] = self.order_list[int(act_params[2])].position[:]
            index_pro, n_pro= map(int, act_params[3:])
            self.drone_list[i_drone].products[index_pro] -= n_pro
            self.drone_list[i_drone].load -= self.catalogue[index_pro]*n_pro
            is_done = self.order_list[int(act_params[2])].deliver(index_pro)
            # if is_done:
            #     self.reward += (self.max_turn - self.turn)/float(self.max_turn)


def main():
    import dimension_reduction as dr
    import tensorflow as tf

    warehouse_list, order_list, drone_list, product_catalogue = pr.parse('busy_day.in')
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
    autoen = dr.autoencoder(400, 2)
    autoen.pose_encoder(2)

    sess.run(tf.global_variables_initializer())
    autoen.train(sess, 1, data, 0.001)

    autoen.pose_train(sess, 10, pose_data, 0.001)

    for i in xrange(100):
        net_out = np.random.normal(size=47)
        print action_converter(net_out, autoen, sess, warehouse_list, order_list, drone_list, 200, product_catalogue, mean, std, pose_mean, pose_std)

if __name__ == "__main__":
    main()
