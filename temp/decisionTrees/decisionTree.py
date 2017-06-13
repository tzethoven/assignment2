import sys
import os
import csv
import pprint
from math import log2

# Header of the table( Sepal Length, Sepal Width, Petal Length, Petal Width, Species)

class DecisionTreeProgram:
    trainingFile="./iris_learn.csv"
    testingFile="./iris.test.csv"

    def __init__(self):
        super().__init__()
        self.ppObject = pprint.PrettyPrinter(indent=0)
        self.classes = set()    # This variable will be transformed as list when readFileCsv is over
        self.setAttribs = []
        for i in range(4):
            self.setAttribs.append(set())
        # self.setAttribs[0].add(1)
        # self.setAttribs[0].add(2)
        # for i in range(3):
        #     self.setAttribs.append([])

        print(self.setAttribs)

        self.table = []
        self.child_table_1 = []
        self.child_table_2 = []
        self.threshold = 0.0
        self.classFrequency = [0,0,0]
        self.selectedAttribute =""
        self.readFileCsv()      # Populate the training table and update classes

        entropia = self.findEntropy(self.table)

        self.splitTable(self.table, self.child_table_1, self.child_table_2,\
                        self.selectedAttribute,\
                        entropia)

        # self.makeTree()


    def splitTable(self, originalTable, table1, table2, attribute, originalEntropy):
        minimum = [99.0,99.0,99.0,99.0]
        maximum = [0.0,0.0,0.0,0.0]


        attributeNumber = len(originalTable[0])-1
        self.bestThreshold = 0
        self.bestInformationGain = 0
        self.bestAttribute = ""
        self.bestTable1 = []
        self.bestTable2 = []

        self.informationGainList = []


        for i in range(len(self.setAttribs)):
            for j, threshold in enumerate(self.setAttribs[i]):
                for k, line in enumerate(originalTable):
                    if line[i] < threshold:
                        table1.append(line)
                    else:
                        table2.append(line)
                
                entropy_table1 = self.findEntropy(table1)
                entropy_table2 = self.findEntropy(table2)

                print("Elementos en Tabla 1: ", len(table1))
                print("Elementos en Tabla 2: ", len(table2))
                print("Elementos en Tabla Original: ", len(originalTable))

                information = (len(table1)/len(originalTable))*entropy_table1 \
                            + (len(table2)/len(originalTable))*entropy_table2
                information_gain = originalEntropy - information

                self.informationGainList.append(information_gain)


                if information_gain > self.bestInformationGain:
                    self.bestInformationGain = information_gain
                    self.bestTable1 = table1
                    self.bestTable2 = table2
                    self.bestThreshold = threshold
                    self.bestAttribute = i


                print("Information gain: ", information_gain)
                print("Table 1 Entropy: ", entropy_table1)

                # self.printTable(table1)
                print("\n")
                print("Table 2 Entropy: ", entropy_table2)
                # self.printTable(table2)
                print("\n\n\n")
                table1 = []
                table2 = []
                entropy_table1 = 0
                entropy_table1 = 0

            print("Gain list of attrib #: ", i)
            print(self.informationGainList)
            self.informationGainList = []

        print("Best information Gain: ", self.bestInformationGain)
        print("Best Attribute #: ", self.bestAttribute)
        print("Best Threshold: ", self.bestThreshold)
        print("\nTable 1:")
        self.printTable(self.bestTable1)
        print("\n\nTable 2:")
        self.printTable(self.bestTable2)


    def findEntropy(self, evaluateTable):
        if evaluateTable:
            self.classFrequency = [0, 0, 0]
            self.entropy = 0.0
            for i, flowerType in enumerate(self.classes):
                for j, sample in enumerate(evaluateTable):
                    if sample[4]==flowerType:
                        self.classFrequency[i] = 1 + self.classFrequency[i]

            for i, value in enumerate(self.classFrequency):
                self.probability = value/sum(self.classFrequency)
                if self.probability == 0:
                    self.entropy = 0.0
                else:
                    self.entropy = self.entropy + self.probability*log2(self.probability)

            self.entropy=-self.entropy
            # print("Calculated Entropy", self.entropy)
            print("Frequencia de clase: ", self.classFrequency)
            return self.entropy
        else:
            self.classFrequency = [0, 0, 0]
            self.entropy = 0.0
            return 999


    def readFileCsv(self):
        with open("iris_learn.csv") as learningData:
            inputFile=csv.reader(learningData)

            for i, data in enumerate(inputFile):
                self.table.append([])
                # print(self.table)
                for j, attribute in enumerate(data):
                    if j < 4 :                          # The first 3 attributes are float numbers
                        self.table[i].append(float(attribute))
                        self.setAttribs[j].add(float(attribute))
                    if j == 4:                          # The last attribute is the class of the flower
                        self.table[i].append(attribute)
                        self.classes.add(attribute)     # Creates a set of Flower Classes set

        self.printTable(self.table)
        self.classes = list(self.classes)
        print(self.setAttribs)

        for i in range(len(self.setAttribs)):
            self.setAttribs[i]=list(self.setAttribs[i])
            self.setAttribs[i].sort()

        print(self.setAttribs)

        print("Classes of the flowers:", self.classes)

    def printTable(self, table_to_print):
        self.ppObject.pprint(table_to_print)


if __name__=="__main__":
    MyTree=DecisionTreeProgram()
    # entropyValue = MyTree.findEntropy()



    # for i, value in enumerate(originalTable):
    #     for j, dimensions in enumerate(value):
    #             if j < 4:                               # Search only in the first 4 columns
    #                 if dimensions > maximum[j]:
    #                     maximum[j] = dimensions
    #                 if dimensions < minimum[j]:
    #                     minimum[j] = dimensions
    #
    # print("Maximum values per column: ", maximum)
    # print("Minimum values per column: ", minimum)