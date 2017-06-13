import csv
import pprint
import math
import copy

class k_Nearest:

    def __init__(self, file_clustered, file_to_cluster, attributes_number, k_near_number):
        self.ppObject = pprint.PrettyPrinter(indent=5, depth=5, width=100)

        self.attributeQty = attributes_number
        self.k_near_number = k_near_number

        self.classes = set()
        self.setAttribs = []
        for i in range(self.attributeQty):
            self.setAttribs.append(set())

        self.table = []                             # Set of objects already Clustered
        self.readFileCsv(file_clustered)            # Populate the training table and update classes variables
        self.table_clustered = copy.deepcopy(self.table) # Save the table in a different variable


        # Reset variables to read again a table to be clustered
        self.classes = set()
        self.setAttribs = []
        for i in range(self.attributeQty):
            self.setAttribs.append(set())

        self.table = []                             # Set of objects to be Clustered
        self.readFileCsv(file_to_cluster)           # Populate the table_to_cluster and update classes variables
        self.table_to_cluster = copy.deepcopy(self.table)

        # self.printTable(table_clustered)
        # print("/////////////////////")
        # self.printTable(table_to_cluster)

        self.k_nearest_cluster()
        self.printTable(self.table_to_cluster)

    def k_nearest_cluster(self):
        table1 = self.table_clustered
        table2 = self.table_to_cluster
        k_near_number = self.k_near_number

        addition = 0
        table3 = copy.deepcopy(table1)

        for i, object_to_cluster in enumerate(table2):
            for j, object_clustered in enumerate(table1):
                for k, coord in enumerate(object_clustered):
                    if k < self.attributeQty:
                        addition = (coord - object_to_cluster[k]) ** 2 + addition
                distance = math.sqrt(addition)
                table3[j].append(distance)
                addition = 0


            table3.sort(key=lambda x: x[-1], reverse=False)

            classes_set = set()
            classes_count = []

            # Save the classes names of the K_near_number of the table3 list
            for m in range(k_near_number):
                classes_set.add(table3[m][-2])
            classes_names = list(classes_set)

            # Count how many class name are in the K_near_number of table 3
            # 1. Select the first cluster/class name
            # 2. Count

            for m, name in enumerate(classes_names):
                classes_count.append([])
                classes_count[m].append(name)

                count=0
                for n in range(k_near_number):
                    if classes_count[m][0] == table3[n][-2]:        # Seach how many are repeated
                        count += 1

                classes_count[m].append(count)
                count=0

            classes_count.sort(key=lambda x: x[-1], reverse=True)

            # self.printTable(classes_count)

            object_to_cluster.append(classes_count[0][0])              # Attach cluster name
            object_to_cluster.append("Confidence: "+ str(classes_count[0][1]/k_near_number))  # Attach cluster name

            table3 = copy.deepcopy(table1)

        return table2

    def readFileCsv(self, training):
        with open(training) as learningData:
            inputFile = csv.reader(learningData)

            for i, data in enumerate(inputFile):
                self.table.append([])
                for j, attribute in enumerate(data):
                    data_range = len(data)
                    if j < self.attributeQty:                       # The first 3 attributes are float numbers
                        self.table[i].append(float(attribute))
                        self.setAttribs[j].add(float(attribute))    # Delete repeating values declaring it as set()
                    if j == data_range-1:                      # The last attribute is the class of the flower
                        self.table[i].append(attribute)
                        self.classes.add(attribute)                 # Creates a set of Flower Classes set

        self.classes = list(self.classes)

    def printTable(self, table_to_print):
        self.ppObject.pprint(table_to_print)

if __name__=="__main__":
    training_file = "./iris_learn.csv"
    testing_file = "./iris_test.csv"
    attribute_number = 4
    k_near_value = 3

    laTabla = k_Nearest(training_file, testing_file, attribute_number, k_near_value)
    algo=laTabla.k_nearest_cluster()
