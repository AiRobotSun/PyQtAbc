from PyQt5 import QtWidgets
from FeatureEngineeringImpl import FeatureEngineeringImpl

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FeatureEngineeringPlatform = FeatureEngineeringImpl()
    FeatureEngineeringPlatform.show()
    sys.exit(app.exec_())
