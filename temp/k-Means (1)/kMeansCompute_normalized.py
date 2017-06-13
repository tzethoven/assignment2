import csv
import pprint
import math
import copy
import random
import matplotlib.pyplot as plt
import time

# Header of the table( Sepal Length, Sepal Width, Petal Length, Petal Width, Species)

class k_Means:

    def __init__(self, training, attributes_number, clusters_number, tolerance, normalize_query):

        self.ppObject = pprint.PrettyPrinter(indent=5, depth=5, width=200)
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

        for i, line in enumerate(self.table):   # Label the original object list to sort at the end of the prog.
            line.append(i+1)

        # self.printTable(self.table)

        self.originalTable = copy.deepcopy(self.table) # Save the originalTable where the results will be plotted

        if normalize_query==True:               # Set the attributes from 0 to 1 scale
            self.normalizeTable()


        for i in range(self.clusterQty):        # Initialize a blank array list for Centroids
            self.newCentroids.append([])
            for j in range(self.attributeQty+1):
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

            if minimum == maximum:      # Meaning that the attribute value repeats all the time
                    maximum = 1
                    minimum = 0

            for j, line in enumerate(self.table):
                line[i] = (line[i]-minimum)/(maximum-minimum)




            # self.printTable(self.table)
            # print(minimum)
            # print(maximum)
            # break
        self.table.sort(key=lambda x: x[-1], reverse=False)
        # self.printTable(self.table)

    def k_cluster(self, method):
        # Select initial values of the centroids:

        if method == 'forgy':
            self.forgyCentroids()
        elif method == 'random':
            self.randomCentroids()
        elif method == 'mao':
            self.maoCentroids(self.table)
        else:
            print("Incorrect clustering method selected")
            exit()

        print("\tInitial values of")
        self.printCentroids(self.centroids)

        print("\tEmpty New Centroid set")
        self.printCentroids(self.newCentroids)

        self.movement = self.compareCentroids(self.centroids, self.newCentroids)    # Displacement of the centroids
                                                                                    # The newCentroids list is 0
        i=1
        print("\nSTARTING CLUSTERING.....................................................")

        while (max([abs(number) for number in self.movement])> self.tolerance):

            tempTable = copy.deepcopy(self.table)                                   # Recover the original data set

            tableClustered = self.distanceAndClustering(tempTable, self.centroids)  # Calc. distance of each object to centroids

            print("\n\tOld centroids")
            self.printCentroids(self.centroids)

            currentCentroids = copy.deepcopy(self.centroids)    # This avoids to change the "old" centroid list

            self.newCentroids = self.calculateCentroids(tableClustered, currentCentroids)    # Move the centroids

            print("\n\tNew centroids")
            self.printCentroids(self.newCentroids)

            self.movement = self.compareCentroids(self.centroids, self.newCentroids)         # Diference between old and new centroids

            self.centroids = copy.deepcopy(self.newCentroids)

            print("Iteration: ", i, " finished. ", "Absolute euclidean distance between old and new centroid: ", self.movement)
            i += 1


        print("Final Centroids Coordinates: ")
        # self.printTable(self.centroids)

        # The end of the clustering:
        # Take the original table (from the .csv file) attaches the cluster number and then return it

        tableClustered.sort(key=lambda x: x[-3], reverse=False)     # Sort it in the same order as the .csv file

        for i, line in enumerate(tableClustered):                   # Copy the final cluster number to the output table
            self.originalTable[i].append(line[-1])

        # self.printData(self.originalTable,self.centroids,7,8)

        self.printAnswerProject(tableClustered,self.centroids)

        # self.originalTable.sort(key=lambda x: x[-2], reverse=False)

        # self.printTable(self.originalTable)

        return self.originalTable

    def maoCentroids(self, sortedObjects):
        print("Euclidean distance from origin was selected as placement method for centroids...")
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

    def forgyCentroids(self):
        # The Forgy method chooses k data points from the dataset at random and uses them as the initial centers.
        print("Forgy selected as placement method for clusters...")

        range_dataset = len(self.table)

        for i in range(self.clusterQty):
            self.centroids.append([])
            rand = random.randint(0, range_dataset-1)   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! check with is the maximum value of range_dataset
            print("Centroid ", i, " placed in Object number: ", rand)
            for j in range(self.attributeQty):
                self.centroids[i].append(self.table[rand][j])

        # self.printTable(self.centroids)

    def randomCentroids(self):
        # The random partition method randomly assigns each data point to one o k partitions, then computes the initial
        # location of the k centers as the mean of the assigned data points
        print("Random Centroids selected as placement method for clusters...")

        for line in self.table:
            line.append(random.randint(0, self.clusterQty-1))         # Append a random cluster to each object
            # print(line[-1])                                                                       ALGO ESTA MAL ACAAAAA REVISAR

        # create and empty array for the list array count
        count = []
        objects_per_cluster_count = []

        for i in range(self.clusterQty):
            count.append([])
            objects_per_cluster_count.append(0.0)           # Array to count how many objects per cluster
            for j in range(self.attributeQty):
                count[i].append(0.0)


        for i, line in enumerate(self.table):
            objects_per_cluster_count[line[-1]] += 1
            for j in range(self.attributeQty):
                count[line[-1]][j] += line[j]               # Sum the values of each attribute for each cluster

        for i, line in enumerate(count):
            for j in range(self.attributeQty):
                line[j] = line[j]/objects_per_cluster_count[i]
            print(objects_per_cluster_count[i], " Objects assigned to Cluster: ", i)

        self.centroids = count

        for line in self.table:                             # Delete the appended ramdom cluster of the data
            del line[-1]

    def calculateCentroids(self, tablero, centroids):

        # centroids = copy.deepcopy(centroide)

        # Create and empty attribute counter
        # Example, attrib=[coord-1 coord-2 coord-3 ... coord-N]  where N is the number of attributes
        attributes = []
        for i in range(self.attributeQty):
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

            print("\tCentroid: ", i,": ", count, " Objetcs in cluster")

            if count > 0:
                centroids[i]=[value/count for value in attributes ]

            count=0

            # Reset the attribute variable
            attributes = []
            for i in range(self.attributeQty):
                attributes.append([])
                attributes[i] = 0.0

        return centroids

    def distanceAndClustering(self, tablero, centroids):

        # Distances to all centroids
        addition=0
        dictionary={}
        for i, coordinates in enumerate(centroids):             # Select each Centroid
            for j, line in enumerate(tablero):                  # Select the object
                for k, data in enumerate(line):                 # Loop through the coordinates
                    if k < self.attributeQty:
                        addition = (coordinates[k]-data) ** 2 + addition    # Euclidean distance
                dictionary.update({"Centroid "+str(i):math.sqrt(addition)}) # Save dist. of the object to the centroid

                if i == 0:
                    tablero[j].append(dictionary)                 # Just to create 1 dictionary per object
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
            data.append(int(nearCentroid))                         # Tag in the last column, the nearest cluster/class
        del dictionary


        # Print how many objects are in each cluster



        return tablero

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
                    else:

                         self.table[i].append(attribute)



                    # if j == self.attributeQty:                                      # The last attribute is the class of the flower
                    #     self.table[i].append(attribute)
                    #     self.classes.add(attribute)                 # Creates a set of Flower Classes set

        self.classes = list(self.classes)

    def printCentroids(self, centroid):
        temp = []

        for i, line in enumerate(centroid):
            temp.append([])
            for value in line:
                temp[i].append('{:.3f}'.format(value))

        print("\tCoordinates: ")
        print("\t-----------------------------------")
        for i, coordinate in enumerate(temp):
            print("\tCentroid: ", i, " -> ", coordinate)
        print("\t-----------------------------------")

    def printTable(self, table_to_print):
        self.ppObject.pprint(table_to_print)

    def printData(self, table_to_figure, centroids_to_figure, attrib1, attrib2):
        x_data=[]
        y_data=[]

        x_centroid = []
        y_centroid = []

        for line in centroids_to_figure:
            x_centroid.append(line[attrib1 - 1])
            y_centroid.append(line[attrib2 - 1])

        for line in table_to_figure:
            x_data.append(line[attrib1-1])
            y_data.append(line[attrib2-1])

        plt.plot(x_data,y_data,'ro')
        plt.plot(x_centroid, y_centroid, 'bo')
        plt.show()

    def printAnswerProject(self, table_to_figure, ctf):
        # Transpose the table

        column = []
        for i in range(self.attributeQty):
            column.append([])

        for i, line in enumerate(table_to_figure):
            for j, value in enumerate(line):
                if j < self.attributeQty:
                    column[j].append(value)

        plt.figure(1)

        # linear
        plt.subplot(331)
        plt.plot(column[0], column[1], 'rx')
        plt.plot(ctf[0][0], ctf[0][1], 'bo')
        plt.plot(ctf[1][0], ctf[1][1], 'go')
        plt.plot(ctf[2][0], ctf[2][1], 'co')
        plt.plot(ctf[3][0], ctf[3][1], 'mo')
        plt.grid(True)

        plt.subplot(332)
        plt.plot(column[2], column[3], 'rx')
        plt.plot(ctf[0][2], ctf[0][3], 'bo')
        plt.plot(ctf[1][2], ctf[1][3], 'go')
        plt.plot(ctf[2][2], ctf[2][3], 'co')
        plt.plot(ctf[3][2], ctf[3][3], 'mo')
        plt.grid(True)

        plt.subplot(333)
        plt.plot(column[4], column[5], 'ro')
        plt.plot(ctf[0][4], ctf[0][5], 'bo')
        plt.plot(ctf[1][4], ctf[1][5], 'go')
        plt.plot(ctf[2][4], ctf[2][5], 'co')
        plt.plot(ctf[3][4], ctf[3][5], 'mo')
        plt.grid(True)

        plt.subplot(334)
        plt.plot(column[6], column[7], 'ro')
        plt.plot(ctf[0][6], ctf[0][7], 'bo')
        plt.plot(ctf[1][6], ctf[1][7], 'go')
        plt.plot(ctf[2][6], ctf[2][7], 'co')
        plt.plot(ctf[3][6], ctf[3][7], 'mo')
        plt.grid(True)

        plt.subplot(335)
        plt.plot(column[8], column[9], 'ro')
        plt.plot(ctf[0][8], ctf[0][9], 'bo')
        plt.plot(ctf[1][8], ctf[1][9], 'go')
        plt.plot(ctf[2][8], ctf[2][9], 'co')
        plt.plot(ctf[3][8], ctf[3][9], 'mo')
        plt.grid(True)

        plt.subplot(336)
        plt.plot(column[10], column[11], 'ro')
        plt.plot(ctf[0][10], ctf[0][11], 'bo')
        plt.plot(ctf[1][10], ctf[1][11], 'go')
        plt.plot(ctf[2][10], ctf[2][11], 'co')
        plt.plot(ctf[3][10], ctf[3][11], 'mo')
        plt.grid(True)

        plt.subplot(337)
        plt.plot(column[12], column[13], 'ro')
        plt.plot(ctf[0][12], ctf[0][13], 'bo')
        plt.plot(ctf[1][12], ctf[1][13], 'go')
        plt.plot(ctf[2][12], ctf[2][13], 'co')
        plt.plot(ctf[3][12], ctf[3][13], 'mo')
        plt.grid(True)

        plt.subplot(338)
        plt.plot(column[14], column[15], 'ro')
        plt.plot(ctf[0][14], ctf[0][15], 'bo')
        plt.plot(ctf[1][14], ctf[1][15], 'go')
        plt.plot(ctf[2][14], ctf[2][15], 'co')
        plt.plot(ctf[3][14], ctf[3][15], 'mo')
        plt.grid(True)

        plt.subplot(339)
        plt.plot(column[16], column[17], 'ro')
        plt.plot(ctf[0][16], ctf[0][17], 'bo')
        plt.plot(ctf[1][16], ctf[1][17], 'go')
        plt.plot(ctf[2][16], ctf[2][17], 'co')
        plt.plot(ctf[3][16], ctf[3][17], 'mo')
        plt.grid(True)



        plt.show()



if __name__=="__main__":
    # training_file = "./iris_learn.csv"
    training_file = "./measurements_learn.csv"
    # training_file = "./measurements_learn2.csv"    # 14 Attributes
    testing_file = "./iris_test.csv"
    attribute_number = 18
    cluster_number = 4
    tolerance = 0.01
    normalize = False

    # Method 1: 'forgy' for Forgy method
    # Method 2: 'random' for random partition
    # Method 3: 'mao' for euclidean distance

    laTabla = k_Means(training_file, attribute_number, cluster_number, tolerance, normalize)
    table_final = laTabla.k_cluster('mao')
    # laTabla.printTable(table_final)