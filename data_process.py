import tensorflow as tf
import numpy as np

class warehouse():
    def __init__(self, position, products):
        self.products = products
        self.position = position

class order():
    def __init__(self, position, products, n_products):
        self.position = position
        self.products = [0 for i in xrange(n_products)]
        for i in products:
            self.products[i] += 1
        self.undelivered_products = sum([0 if i == 0 else 1 for i in self.products])
    def deliver(index_product):
        self.products[index_product] = 0
        if self.undelivered_products > 1:
            self.undelivered_products -= 1
            return 0
        else:
            self.undelivered_products -= 1
            return 1


class drone():
    def __init__(self, position, n_products):
        self.position = position
        self.load = 0
        self.products = [0 for i in xrange(n_products)]
