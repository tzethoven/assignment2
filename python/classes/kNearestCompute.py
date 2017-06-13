import csv
import pprint
import math
import copy
# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

class k_Nearest:

    def __init__(self, file_clustered, file_to_cluster, attributes_number, k_near_number, tableInput, tableSorted, tableTesting):
        self.ppObject = pprint.PrettyPrinter(indent=5, depth=5, width=100)

        self.attributeQty = attributes_number
        self.k_near_number = k_near_number

        # Temporary variables to read the data of the table to be clustered with k-means
        self.classes = set()
        self.setAttribs = []
        for i in range(self.attributeQty):
            self.setAttribs.append(set())

        self.table = []                                  # Temp variable to store table

        if tableInput == False:
            self.readFileCsv(file_clustered)             # Populate the training table if the source is a csv file
        else:
            self.table=copy.deepcopy(tableSorted)        # Populate the training table if the list is already provided

        self.table_clustered = copy.deepcopy(self.table) # Save the table in a different variable


        # Reset temporary variables again to read the data of the table to be clustered
        self.classes = set()
        self.setAttribs = []
        for i in range(self.attributeQty):
            self.setAttribs.append(set())

        self.table = []                                 # Set of objects to be Clustered

        if tableInput == False:
            self.readFileCsv(file_to_cluster)           # Populate the table_to_cluster and update classes variables
        else:
            self.table = copy.deepcopy(tableTesting)

        self.table_to_cluster = copy.deepcopy(self.table)

    def k_nearest_cluster(self, clusterLabeling):

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
                    if classes_count[m][0] == table3[n][-2]:            # Search how many are repeated
                        count += 1

                classes_count[m].append(count)
                count=0

            classes_count.sort(key=lambda x: x[-1], reverse=True)      # Sorting from highest to lowest in the count

            # Merge the clusters with similar labeling

            object_to_cluster.append(classes_count[0][0])              # Attach cluster index (integer)

            ##########################################################################################################
            ##########################################################################################################
            # CONFIDENCE INTERVAL CALCULATION
            ##########################################################################################################
            ##########################################################################################################


            table4 = copy.deepcopy(table3)

            # Replace the table 4 with the labelings already sorted by the Bus States

            for i,row in enumerate(table4):
                row[-2]=clusterLabeling[row[-2]]

            # Run again everything
            classes_set = set()
            classes_count = []

            for m in range(k_near_number):
                classes_set.add(table4[m][-2])
            classes_names = list(classes_set)

            for m, name in enumerate(classes_names):
                classes_count.append([])
                classes_count[m].append(name)

                count = 0
                for n in range(k_near_number):
                    if classes_count[m][0] == table4[n][-2]:  # Seach how many are repeated
                        count += 1

                classes_count[m].append(count)
                count = 0

            classes_count.sort(key=lambda x: x[-1], reverse=True)  # Sorting from highest to lowest in the count

            # Merge the clusters with similar labeling

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

    def printSolution(self, object_index, clusterLabels):
        clusterLabeling = copy.deepcopy(clusterLabels)
        # self.table_clustered has the cluster number at the -1 position of the list (Training Data Set)
        # self.table_to_cluster is the Test data set already with the cluster number in the -2 position


        left = 0.06  # the left side of the subplots of the figure
        right = 0.95  # the right side of the subplots of the figure
        bottom = 0.13  # the bottom of the subplots of the figure
        top = 0.95  # the top of the subplots of the figure
        wspace = 0.27  # the amount of width reserved for blank space between subplots
        hspace = 0.27  # the amount of height reserved for white space between subplots


        colors = ['bo','go','co','mo','yo','ko','bx','gx','cx','mx','yx','kx','wx','wo']
        auxiliary = [0,2,4,6,8,10,12,14,16]
        subplot_index = [331,332,333,334,335,336,337,338,339]

        # Transpose the Training table already clustered
        column = []
        for i in range(self.attributeQty):
            column.append([])

        for i, line in enumerate(self.table_clustered):
            for j, value in enumerate(line):
                if j < self.attributeQty:
                    column[j].append(value)


        # Transpose the Sorted table
        column2 = []
        for i in range(self.attributeQty):
            column2.append([])

        for i, line in enumerate(self.table_to_cluster):
            for j, value in enumerate(line):
                if j < self.attributeQty:
                    column2[j].append(value)

        plt.close('all')
        plt.figure(1)
        plt.suptitle("OBJECT #" + str(object_index+1) + " Classified as \"" + str(clusterLabels[self.table_to_cluster[object_index][-2]])
                     + "\" with " + self.table_to_cluster[object_index][-1])

        labeling = []
        for i, row in enumerate(clusterLabeling):
            labeling.append([])
            labeling[i].append(row)
            labeling[i].append(0)

        # Print the Training table set with the Colors of its respective cluster
        for j, row in enumerate(self.table_clustered):
            for k, i in enumerate(auxiliary):
                ax = plt.subplot(subplot_index[k])

                if labeling[int(row[-1])][1]==0:
                    label_string = clusterLabels[int(row[-1])]
                else:
                    label_string = ""

                ax.plot(column[i][j], column[i + 1][j], colors[int(row[-1])], label=label_string)
                labeling[int(row[-1])][1]=1;

                ax.set_xlabel(self.busName(i))
                ax.set_ylabel(self.busName(i+1))
                ax.grid(True)

                plt.legend(bbox_to_anchor=(0, -3, 1.5, 1), loc='lower center', ncol=5, mode='expand')


        # Print the object of the Sorted table
        for k, i in enumerate(auxiliary):
            ax1 = plt.subplot(subplot_index[k])
            ax1.plot(self.table_to_cluster[object_index][i],self.table_to_cluster[object_index][i+1],"rx",markersize=10,label="Data Object" if k==0 else "")
            plt.legend(bbox_to_anchor=(0, -3, 3, 3), loc='lower center', ncol=6, mode='expand')


        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        plt.show()

    def busName(self, variable):
        busNames = {"AMHE_ANG": 0,
                    "AMHE_VOLT": 1,
                    "BOWM_ANG":2,
                    "BOWM_VOLT":3,
                    "CLAR_ANG":4,
                    "CLAR_VOLT":5,
                    "CROSS_ANG":6,
                    "CROSS_VOLT":7,
                    "GRAN_ANG":8,
                    "GRAN_VOLT":9,
                    "MAPL_ANG":10,
                    "MAPL_VOLT":11,
                    "TROY_ANG":12,
                    "TROY_VOLT":13,
                    "WAUT_ANG":14,
                    "WAUT_VOLT":15,
                    "WINL_ANG":16,
                    "WINL_VOLT":17
        }
        if isinstance(variable, str):
            return busNames[variable]
        else:
            return list(busNames.keys())[list(busNames.values()).index(variable)]

    def processTable(self, centroidLabels):
        tempTable=copy.deepcopy(self.table_to_cluster)

        for i in range(len(tempTable)):
            tempTable[i][-2] = centroidLabels[tempTable[i][-2]]

        return tempTable

if __name__=="__main__":
    training_file = "./iris_learn.csv"
    testing_file = "./iris_test.csv"
    attribute_number = 4
    k_near_value = 3

    tableInput = False  # Selected if the table comes from SQL database and not .csv file
    tableSorted =[]
    tableTesting = []

    laTabla = k_Nearest(training_file, testing_file, attribute_number, k_near_value, tableInput, tableSorted, tableTesting)
    laTablaFinal=laTabla.k_nearest_cluster()

    object_index = 10
    laTabla.printSolution(laTabla.table_clustered,laTablaFinal, object_index)
