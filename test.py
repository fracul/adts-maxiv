#!/opt/local/bin/python
import testplots
from matplotlib import pyplot

def allPaperPlots():
    """
    Profce all plots in the manuscript and also test all aspects of the code
    """

    twoMode = testplots.prepCompPlot(twoMode=True)
    testplots.landauContours(twoMode[1],signb=1,csinv=True)
    testplots.landauContours(twoMode[1],signb=-1,csinv=True)

    delphBMark = testplots.prepDelphiBMark()
    testplots.delphiCompareSuite(*delphBMark)

def main():
    allPaperPlots()
    pyplot.show()

if __name__=="__main__":
    main()
