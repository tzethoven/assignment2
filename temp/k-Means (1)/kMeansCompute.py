import sys
import os
import csv
import pprint
import math
import copy
import time

# Header of the table( Sepal Length, Sepal Width, Petal Length, Petal Width, Species)

class k_Means:

    def __init__(self, training, attributes_number, clusters_number, tolerance, normalize_query):

        self.ppObject = pprint.PrettyPrinter(indent=5, depth=5, width=100)
        self.classes = set()    # This variable will be transformed as list when readFileCsv is over
        self.setAttribs = []
        self.centroids = []
        self.newCentroids = []


        self.attributeQty = attributes_number   # Number of attributes per object
        self.clusterQty = clusters_number       # This is the number of cluster we want to make (same value as classes)

        self.tolerance = tolerance

        for i in range(self.attributeQty):
            self.setAttribs.append(set())

        self.table = []                         # Set of objects to be Clustered
        self.readFileCsv(training)              # Populate the training table and update classes variables
        # self.printTable(self.table)

        self.originalTable = copy.deepcopy(self.table) # Save the originalTable where the results will be plotted

        if normalize_query==True:
            self.normalizeTable()

        for i in range(self.clusterQty):        # Initialize array list of Centroids
            self.newCentroids.append([])
            for j in range(len(self.table[0])+1):
                self.newCentroids[i].append([])
                self.newCentroids[i][j]=0.0










    def normalizeTable(self):
        # This function will normailze the self.Table
        # If the function is not executed, the data will be clustered with its original values

        min_max_attr = []

        for i in range(self.attributeQty):
            min_max_attr.append([])
            min_max_attr[i].append("Attribute "+ str(i) + ":")
            self.table.sort(key=lambda x: x[i], reverse=False)
            minimum = self.table[0][i]
            maximum = self.table[-1][i]

            if minimum == maximum:
                minimum =0




            self.printTable(self.table)
            print(minimum)
            print(maximum)
            break



    def k_cluster(self):
        # Select initial values of the centroids:
        # We will select the nearest and far points with respect to origin (as minimum 2 classes - 2 clusters)
        # Then add the more centroids as required (segment division)
        # self.centroid is populated as required

        self.initialCentroids(self.table)
        self.movement = self.compareCentroids(self.centroids, self.newCentroids)    # Displacement of the centroids

        i=1
        print("Initial guess of Centroids: ")

        while (max([abs(number) for number in self.movement])> self.tolerance):
            tempTable = copy.deepcopy(self.table)       # Recover the original data set
            tableClustered = self.distanceAndClustering(tempTable, self.centroids)

            # self.printTable(tableClustered)

            currentCentroids = copy.deepcopy(self.centroids)

            self.newCentroids = self.calculateCentroids(tableClustered, currentCentroids)    # ////////////////////////////////////////////////////////////////////////////////


            self.movement = self.compareCentroids(self.centroids, self.newCentroids)
            self.centroids = copy.deepcopy(self.newCentroids)

            print("Iteration: ", i, " distance movement: ", self.movement)
            i += 1

        # self.printTable(tableClustered)
        # myList = [['%.2f' % i for i in linea] for linea in self.centroids ]
        print("Final Centroids Coordinates: ")
        # self.printTable(self.centroids)

        return tableClustered




    def calculateCentroids(self, tablero, centroids):
        # Create and empty attribute counter
        # Example, attrib=[coord-1 coord-2 coord-3 ... coord-N]  where N is the number of attributes
        attributes = []
        for i in range(len(self.table[0])):
            attributes.append([])
            attributes[i] = 0.0


        # Sort by Centroid number (last column of the table)

        tablero.sort(key=lambda x: x[-1], reverse=False)
        count=0

        for i, coordenadas in enumerate(centroids):
            for j, line in enumerate(tablero):
                if line[-1] == i:                           # Index -i- its the same as cluster number
                    count += 1
                    for k, data in enumerate(line):
                        if k < self.attributeQty:
                            attributes[k]=attributes[k]+data

            print("\t\tCentroid: ", i,": ", count, " Objetcs in cluster")

            if count > 0:
                centroids[i]=[value/count for value in attributes ]

            count=0

            # Reset the attribute variable
            attributes = []
            for i in range(len(self.table[0])):
                attributes.append([])
                attributes[i] = 0.0

        return centroids

    def distanceAndClustering(self, tablero, centroids):

        # Distances to all centroids
        addition=0
        dictionary={}
        for i, coordinates in enumerate(centroids):             # Select each Centroid
            for j, line in enumerate(tablero):                    # Select the object
                for k, data in enumerate(line):                 # Loop through the coordinates
                    if k < self.attributeQty:
                        addition = (coordinates[k]-data) ** 2 + addition    # Euclidean distance
                dictionary.update({"Centroid "+str(i):math.sqrt(addition)}) # Save dist. of the object to the centroid

                if i == 0:
                    tablero[j].append(dictionary)                 # Just to create 1 dictionary
                else:
                    tablero[j][-1].update(dictionary)             # Add the additional centroids to the dictionary

                dictionary = {}
                addition=0

        # Cluster the objects depending on the distances
        del dictionary
        for i, data in enumerate(tablero):
            dictionary=data[-1]
            values = sorted(dictionary, key=dictionary.__getitem__)
            nearCentroid = values[0][-1]
            data.append(int(nearCentroid))
        del dictionary
        # self.printTable(tablero)
        return tablero



        # self.printTable(tablero)
        # self.printTable(table[20][5].items())

    def compareCentroids(self, old, new):
        # print("Old centroid values: ")
        # self.printTable(old)
        # print("New Centroid values")
        # self.printTable(new)

        # time.sleep(3)
        movement=[]
        for i, line in enumerate(old):
            movement.append([])
            for j, value in enumerate(line):
                if j < self.attributeQty:
                    movement[i].append(old[i][j]-new[i][j])

        for i, data in enumerate(movement):
            movement[i]=sum(data)


        return movement

    def initialCentroids(self, sortedObjects):

        # Distance of the object attributes with respect to origin
        for i, data in enumerate(sortedObjects):
            squared = list(map(lambda x: x ** 2, data[0:self.attributeQty]))
            sortedObjects[i].append(math.sqrt(sum(squared)))

        # Sort by the distance from nearest to farthest with respect to origin
        sortedObjects.sort(key=lambda x:x[-1], reverse=False)


        # Minimum 2 classes added, meaning 2 centroids
        self.centroids.append(sortedObjects[0])
        self.centroids.append(sortedObjects[-1])

        # More clusters added, adding more centroids
        steps = int(len(sortedObjects)/(self.clusterQty-1))
        count=0
        for i in range(steps,len(sortedObjects), steps):
            if count < self.clusterQty-2:
                self.centroids.append(sortedObjects[i])
            count += 1

        for i, line in enumerate(sortedObjects):
            del line[-1]

    def readFileCsv(self, training):
        with open(training) as learningData:
            inputFile = csv.reader(learningData)

            for i, data in enumerate(inputFile):
                self.table.append([])
                # print(self.table)
                for j, attribute in enumerate(data):
                    if j < self.attributeQty:                                       # The first 3 attributes are float numbers
                        self.table[i].append(float(attribute))
                        self.setAttribs[j].add(float(attribute))    # Delete repeating values declaring it as set()
                    # if j == self.attributeQty:                                      # The last attribute is the class of the flower
                    #     self.table[i].append(attribute)
                    #     self.classes.add(attribute)                 # Creates a set of Flower Classes set

        self.classes = list(self.classes)

    def printTable(self, table_to_print):
        self.ppObject.pprint(table_to_print)

if __name__=="__main__":
    # training_file = "./iris_learn.csv"
    training_file = "./measurements_learn.csv"
    testing_file = "./iris_test.csv"
    attribute_number = 18
    cluster_number = 4
    tolerance = 0.01
    normalize = False

    laTabla = k_Means(training_file, attribute_number, cluster_number, tolerance, normalize)
    table_final = laTabla.k_cluster()
    # laTabla.printTable(table_final)