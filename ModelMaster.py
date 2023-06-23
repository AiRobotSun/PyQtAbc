from PyQt5 import QtWidgets
from MachineLearningToolImpl import MachineLearningToolImpl

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MachineLearningPlatform = MachineLearningToolImpl()
    MachineLearningPlatform.show()
    sys.exit(app.exec_())
