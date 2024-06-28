import testplots

def allPaperPlots():
    """
    Profce all plots in the manuscript and also test all aspects of the code
    """

    twoMode = testplots.prepCompPlot(returnFirst=False,twoMode=True)
    testplots.landauContours(twoMode[1],signb=1,csinv=True)
    testplots.landauContours(twoMode[1],signb=-1,csinv=True)

    delphBMark = testplots.prepDelphiBMark()
    testplots.delphiCompareSuite(*delphBMark)
