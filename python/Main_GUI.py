import classes
from classes.SQL import SQL
from classes.kMeansCompute import k_Means
from classes.kNearestCompute import k_Nearest



import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QApplication, QFileDialog, QSizePolicy

# from matplotlib.backends.backend_qt5agg import
# from matplotlib.figure import Figure
# from matplotlib.lines import Line2D
# import matplotlib.pyplot as plt
#
# import random


Ui_MainWindow, QtBaseClass = uic.loadUiType("Qt/GUI.ui")

class Main_GUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.initial_vals()
        self.update_btns()
        self.define_event_handlers()

    def initial_vals(self):
        self.edt_sql_port.setText("3306")
        self.edt_sql_user.setText("root")
        self.edt_sql_pw.setText("")

        self.spin_km_k.setValue(4)
        self.spin_knn_k.setValue(4)

        self.radio_km_forgy.setChecked(True)

        self.combo_km1.addItems([])
        self.combo_km2.addItems([])
        self.combo_knn.addItems([])

        # self.edt_km_log.setText("")
        # self.edt_knn_log.setText("")

        self.sql_path = ""
        self.learningData = False
        self.testingData = False
        self.km = False
        self.knn = False
        self.clusters = False




    def update_btns(self):
        self.btn_sql_select.setEnabled(True)
        if self.sql_path:
            self.btn_dl_db.setEnabled(True)
        else:
            self.btn_dl_db.setEnabled(False)

        if self.learningData and self.testingData:
            self.btn_km_run.setEnabled(True)
        else:
            self.btn_km_run.setEnabled(False)

        if self.clusters:
            self.btn_knn_run.setEnabled(True)
        else:
            self.btn_knn_run.setEnabled(False)

        if self.km:
            self.btn_km_plot.setEnabled(True)
            self.btn_km_plot_all.setEnabled(True)
        else:
            self.btn_km_plot.setEnabled(False)
            self.btn_km_plot_all.setEnabled(False)

        if self.knn:
            self.btn_knn_plot.setEnabled(True)
        else:
            self.btn_knn_plot.setEnabled(False)


    def define_event_handlers(self):
        self.btn_sql_select.clicked.connect(self.select_sql)
        self.btn_dl_db.clicked.connect(self.download_db)

        self.btn_km_run.clicked.connect(self.run_kmean)
        self.btn_km_plot.clicked.connect(self.plot_kmean)
        self.btn_km_plot_all.clicked.connect(self.plot_all_kmean)
        self.btn_knn_run.clicked.connect(self.run_knn)
        self.btn_knn_plot.clicked.connect(self.plot_knn)

    def sql_error(self, e):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("SQL error")
        msg.setText("Failed to process SQL file, please make sure to select a correct file and enter your correct"
                    "credentials.")
        msg.setDetailedText(str(e))
        msg.exec_()

    def print_data(self, datas, tables, headers=None, label_names=None, tabs=None, kn=None):
        for data, table in zip(datas, tables):
            table.setRowCount(len(data))
            table.setColumnCount(len(data[0]))
            for i, row in enumerate(data):
                for j, element in enumerate(row):
                    table.setItem(i, j, QtWidgets.QTableWidgetItem("{}".format(element)))

        if tabs:
            for i in range(len(tabs)):
                self.tabs_sql.setTabText(i, tabs[i])

        if headers:
            for combo in [self.combo_km1, self.combo_km2]:
                combo.clear()
                combo.addItems(headers)

        if label_names:
            tables[0].setVerticalHeaderLabels(label_names)
            tables[0].setHorizontalHeaderLabels(self.headers)

        if kn:
            tables[0].setHorizontalHeaderLabels(kn + self.headers)

        self.update_btns()

    def select_sql(self):
        self.sql_path, _ = QFileDialog.getOpenFileName(self, caption="Please select an SQL file", filter="*.sql")
        self.update_btns()

    def download_db(self):
        try:
            port = int(self.edt_sql_port.text())
            user = self.edt_sql_user.text()
            pw = self.edt_sql_pw.text()

            db = SQL(port=port,
                     user=user,
                     pw=pw,
                     sql_path=self.sql_path)
        except Exception as e:
            self.sql_error(e)
            return

        self.learningData, self.headers = db.get_table("measurements")
        self.testingData, _ = db.get_table("analog_values")
        db.close_sql()

        self.print_data(datas=[self.learningData, self.testingData],
                        tables=[self.table_sql1, self.table_sql2],
                        headers=self.headers,
                        tabs=["measurements", "analog_values"])

    def run_kmean(self):
        k = self.spin_km_k.value()
        method = [radio.isChecked() for radio in [self.radio_km_forgy, self.radio_km_random, self.radio_km_euclyd]]
        method = [m for b, m in zip(method, ["forgy", "random", "euclidean"]) if b][0]
        try:
            self.km = k_Means(training="",
                              attributes_number=18,
                              clusters_number=k,
                              tolerance=0.01,
                              normalize_query=False,
                              tableInput=True,
                              tableData=self.learningData)
            self.clusters = self.km.k_cluster(method)

            self.print_data([self.km.centroids], [self.table_km], label_names=self.km.clusterLabeling)
        except Exception as e:
            print(e)

    def plot_kmean(self):
        x1 = self.combo_km1.currentText()
        x2 = self.combo_km2.currentText()
        self.km.printVariables(x1, x2, self.clusters, self.km.centroids)

    def plot_all_kmean(self):
        self.km.printAnswerProject(self.km.table_Clustered, self.km.centroids)

    def run_knn(self):
        k = self.spin_knn_k.value()
        try:
            self.knn = k_Nearest(file_clustered="",
                                 file_to_cluster="",
                                 attributes_number=18,
                                 k_near_number=k,
                                 tableInput=True,
                                 tableSorted=self.clusters,
                                 tableTesting=self.testingData)
            self.knn.k_nearest_cluster(self.km.clusterLabeling)
            self.knn_table = self.knn.processTable(self.km.clusterLabeling)
            self.knn_table = [row[-2:] + row[:-2] for row in self.knn_table]
            self.print_data([self.knn_table], [self.table_knn], kn=["CLASS", "CONF. INTERVAL"])
            itms = [str(x + 1) for x in range(len(self.knn_table))]
            self.combo_knn.clear()
            self.combo_knn.addItems(itms)
        except Exception as e:
            print(e)

    def plot_knn(self):
        try:
            x = int(self.combo_knn.currentText()) - 1

            self.knn.printSolution(x, self.km.clusterLabeling)
        except Exception as e:
            print(e)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main_GUI()
    window.show()
    sys.exit(app.exec_())
