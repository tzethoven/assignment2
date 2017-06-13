import classes
from classes.SQL import SQL
from classes.kMeansCompute import k_Means
from classes.kNearestCompute import k_Nearest
import pprint



if __name__ == "__main__":
    sql = SQL(sql_path=r"..\assignment2code_test.sql")
    sql.get_table("measurements")
    learningData = sql.data
    sql.get_table("analog_values")
    trainingData = sql.data
    sql.close_sql()

    pprintObject = pprint.PrettyPrinter(indent=5, depth=5, width=400)
    print("Learning data objects: ", len(learningData))
    print("Learning data attributes: ", len(learningData[0]))
    print("Training data objects: ", len(trainingData))
    print("Training data attributes: ", len(trainingData[0]))
    # pprintObject.pprint(trainingData)



    # ------------------------------------- Settings for the Algorithms--------------------------------------------
    # training_file = "./iris_learn.csv"
    training_file = "./measurements_learn.csv"
    # training_file = "./measurements_learn2.csv"    # 14 Attributes
    testing_file = "./iris_test.csv"
    attribute_number = 18
    cluster_number = 5
    tolerance = 0.01
    normalize = False

    tableInput = True  # Selected True, if the table comes from SQL database and not a .csv file

    # Method 1: - '100' - 'forgy' for Forgy method
    # Method 2: - '010' - 'random' for random partition
    # Method 3: - '001' - 'euclidean' for euclidean distance

    # ------------------------------------- k- Means Algorithm ----------------------------------------------------
    laTabla = k_Means(training_file, attribute_number, cluster_number, tolerance, normalize, tableInput, learningData)
    tableSorted = laTabla.k_cluster("euclidean")

    # laTabla.printVariables("BOWM_ANG", "BOWM_VOLT", tableSorted, laTabla.centroids)


    # laTabla.printAnswerProject(laTabla.table_Clustered, laTabla.centroids)

    # print("\nTable Sorted:")
    # laTabla.printTable(tableSorted)

    # ------------------------------------- k- Nearest Algorithm ----------------------------------------------------
    # Method for the k-nearest algorithm
    k_near_value = 7
    tableTesting = trainingData
    object_index = 15               # The object which I want to check if the clustering is ok

    k_NearestObject = k_Nearest(training_file, testing_file, attribute_number, k_near_value, tableInput, tableSorted,
                        tableTesting)

    laTablaFinal = k_NearestObject.k_nearest_cluster(laTabla.clusterLabeling)       # Table sorted with the final solution


    k_NearestObject.printSolution(object_index, laTabla.clusterLabeling)

    tableToPrint = k_NearestObject.processTable(laTabla.clusterLabeling)            # LAB

    # print("\n\n\n\nSORTED TABLE:")
    # pprintObject.pprint(tableToPrint)

