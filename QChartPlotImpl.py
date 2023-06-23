from PyQt5.QtChart import QChart, QValueAxis, QSplineSeries
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QPainter, QBrush


# 动态图绘制类
class QChartViewPlot(QChart):
    def __init__(self, parent=None):
        super(QChartViewPlot, self).__init__(parent)
        self.zRangeMax = None
        self.zRangeMin = None
        self.curve_name = None
        self.series = None
        self.seriesPool = []
        self.seriesCount = 0
        self.axisY = None
        self.axisX = None
        self.axisZ = None
        self.yRangeMax = 100
        self.yRangeMin = 0
        self.xRangeMax = 1024
        self.xRangeMin = 0
        self.counter = 0
        self.window = parent
        self.legend().show()

    def set_xrange(self, min_x=0, max_x=10000):
        self.xRangeMax = max_x
        self.xRangeMin = min_x
        self.axisX.setRange(self.xRangeMin, self.xRangeMax)
        print("[QChart] set x range: [{}, {}] done.".format(min_x, max_x))

    def set_yrange(self, min_y=0, max_y=100):
        self.yRangeMin = min_y
        self.yRangeMax = max_y
        self.axisY.setRange(self.yRangeMin, self.yRangeMax)
        print("[QChart] set y range: [{}, {}] done.".format(min_y, max_y))

    def add_xy_axis(self):
        print("[QChart] add xy axis enter.")
        self.axisX = QValueAxis()
        self.addAxis(self.axisX, Qt.AlignBottom)
        self.axisY = QValueAxis()
        self.addAxis(self.axisY, Qt.AlignLeft)
        print("[QChart] add xy axis done.")

    def add_series(self, name="Curve"):
        print("[QChart] add series 2d enter.")
        self.series = QSplineSeries()
        self.series.setName(name)
        self.series.setUseOpenGL(True)
        self.addSeries(self.series)
        self.series.attachAxis(self.axisX)
        self.series.attachAxis(self.axisY)
        self.curve_name = name
        print("[QChart] add series 2d done.")

    def add_multiple_series(self, name="Curve", color=QPen(Qt.black)):
        print("[QChart] add series 2d enter.")
        series = QSplineSeries()
        series.setName(name)
        series.setUseOpenGL(True)
        series.setPen(color)
        self.addSeries(series)
        series.attachAxis(self.axisX)
        series.attachAxis(self.axisY)
        self.seriesPool.append(series)
        self.seriesCount += 1
        print("[QChart] add series to pool done.")
        return self.seriesCount - 1

    def clearAllSeries(self):
        self.seriesCount = 0
        self.seriesPool.clear()
        self.removeAllSeries()
        print("[QChart] Remove all series done.")

    # 逐个更新y的值
    def handle_update(self, ydata):
        print("[QChart] Update data for {} enter: {}.".format(self.curve_name, self.counter))
        # x个数够则直接放到后面，不够，则刷新x的坐标个数
        if self.counter >= self.xRangeMax:
            self.set_xrange(self.xRangeMin, self.xRangeMax+10)
        # 更新y轴坐标范围
        print("[QChart] Update y axis range.")
        points = self.series.pointsVector()
        if self.counter <= 5:
            self.series.append(self.counter, ydata)
            self.counter += 1
            return
        for i in range(self.counter - 1):
            points[i].setY(points[i + 1].y())
        y_min = min(points, key=lambda point: point.y()).y()
        y_max = max(points, key=lambda point: point.y()).y()
        self.set_yrange(y_min*0.9, y_max*1.1)
        self.series.append(self.counter, ydata)
        self.counter += 1
        print("[QChart] Update data for {} done.".format(self.curve_name))

    def draw(self, xdata, ydata, seriesIndex):
        # 直接整个的将xy曲线绘制, xdata和ydata数据一一对应
        pointNum = len(xdata)
        for i in range(pointNum):
            self.seriesPool[seriesIndex].append(xdata[i], ydata[i])
        print("Update curve data done.")


