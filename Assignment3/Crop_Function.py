import cv2
import matplotlib.pyplot as plt


# cite: class taken from https://stackoverflow.com/questions/28758079/
# python-how-to-get-coordinates-on-mouse-click-using-matplotlib-canvas

class Crop():
    def __init__(self, name):
        self.fname = name
        self.img = cv2.imread(self.fname)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.point = ()

    def getCoord(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(self.img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return self.point

    def __onclick__(self,click):
        self.point = (click.xdata, click.ydata)
        return self.point
