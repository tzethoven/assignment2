import csv
import math
import random
import numpy as np

class K_mean:
    def __init__(self, csv, k):
        flowers = self.process_csv(csv)
        self.learning = random.sample(flowers, 120)
        self.testing = list(set(flowers) - set(self.learning))
        self.k = k
        self.find_kmeans()

        # score = self.test(self.testing)
        # print(score)

    def find_kmeans(self):
        # means = self.find_initial_means_Forgy(k)
        means = self.find_initial_means_partition(self.k)
        self.iterate_means(means)








    def iterate_means(self, means):
        for flower in self.learning:
            point = self.flower_to_point(flower)
            d = []
            for i in range(self.k):
                d.append(self.calc_distance(point, means[i]))
            np.argmin(d)



    def calc_distance(self, p, m):
        d2 = 0
        for i in range(len(p)):
            d2 += (p[i] - m[i])**2
        return math.sqrt(d2)

    def find_initial_means_Forgy(self, k):
        flowers = random.sample(self.learning, k)
        means = []
        for flower in flowers:
            a, b, c, d, _ = flower
            means.append((a, b, c, d,))
        return means

    def find_initial_means_partition(self, k):
        parts = self.partitions(k)
        means = []
        for part in parts.values():
            n = len(part)
            a_sum, b_sum, c_sum, d_sum = 0, 0, 0, 0
            for flower in part:
                a, b, c, d, _ = flower
                a_sum += a/n
                b_sum += b/n
                c_sum += c/n
                d_sum += d/n
            means.append((a_sum, b_sum, c_sum, d_sum,))
        return means

    def partitions(self, k):
        lst = self.learning
        random.shuffle(lst)
        partitions = {}
        for i in range(k):
            partitions[i] = [lst.pop()]
        for flower in lst:
            p = random.randint(0, k-1)
            partitions[p].append(flower)
        return partitions

    @staticmethod
    def flower_to_point(flower):
        a, b, c, d, _ = flower
        return (a, b, c, d,)

    @staticmethod
    def process_csv(csv_file):
        flowers = []
        with open(csv_file, "rt") as file:
            reader = csv.reader(file)
            for row in reader:
                tup = (float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4])
                flowers.append(tup)
        return flowers


if __name__ == "__main__":
    K_mean("iris_2.csv", 5)