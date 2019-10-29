import numpy as np
import matplotlib.pyplot as plt
import image_matcher_eval as ime
import pandas as pd
from tabulate import tabulate

if __name__ == '__main__':
    data = []
    summary = []
    methodId = 1
    name_ref = "darts1_1"

    name_perspectives = ['darts2_1', 'darts_alul', 'darts_bal', 'darts_felul', 'darts_jobb']
    minHamming_prefilters = range(20, 50, 2)
    maxHamming_postfilters = [False]
    cross_Checks = [False]
    maxRatio_postfilters = [False]
    methodNames = ['cvBF']

    for methodName in methodNames:
        for maxRatio_postfilter in maxRatio_postfilters:
            for cross_Check in cross_Checks:
                if (cross_Check and maxRatio_postfilter) or (methodName == 'FLANN' and cross_Check):
                    continue
                for maxHamming_postfilter in maxHamming_postfilters:
                    for minHamming_prefilter in minHamming_prefilters:
                        for name_perspective in name_perspectives:
                            data, summary = ime.runMethod(data=data, summary=summary, methodId=methodId,
                                                          method_name=methodName, name_ref=name_ref,
                                                          name_perspective=name_perspective,
                                                          prefilterValue=minHamming_prefilter, crossCheck=cross_Check,
                                                          postFilterHamming=maxHamming_postfilter,
                                                          postFilterRatio=maxRatio_postfilter,
                                                          outputName=ime.makeMethodName(methodName,
                                                                                        prefilterValue=minHamming_prefilter,
                                                                                        crossCheck=cross_Check,
                                                                                        postFilterHamming=maxHamming_postfilter,
                                                                                        postFilterRatio=maxRatio_postfilter))
                            print(ime.makeMethodName(methodName, prefilterValue=minHamming_prefilter,
                                                     crossCheck=cross_Check,
                                                     postFilterHamming=maxHamming_postfilter,
                                                     postFilterRatio=maxRatio_postfilter) + '_' + name_perspective)

                            if len(summary) == len(name_perspectives):
                                data = ime.addSummaryRow(data, summary)
                                summary.clear()
                                methodId += 1

    headers = ["Id", "Image", "Prefilt", "#Detected", "#Filt",
               "Image", "#Detected", "Det(ms)",
               "Method", "Cross", "Max.Dist.", "Max.Ratio",
               "#Matches", "%Matches", "#Correct", "%Correct",
               "Match(ms)"]

    print(tabulate(data, headers=headers))

    with open("output/PrefilterPlot.txt", 'w', encoding='utf-8') as f:
        f.write(tabulate(data, headers=headers))

    x = minHamming_prefilters
    y = [i[14] for i in data]
    line = plt.plot(x, y, 'g-', x, y, 'g^')

    plt.title('Number of correct match')
    plt.xlabel('Prefilter value')
    plt.ylabel('#Correct')

    plt.savefig('output/#PrefilterPlot.png', bbox_inches="tight")
    plt.show()

    # ---------------------
    y = [i[15] for i in data]

    line = plt.plot(x, y, 'g-', x, y, 'g^')

    plt.title('Percent of correct match')
    plt.xlabel('Prefilter value')
    plt.ylabel('%Correct')

    plt.savefig('output/%PrefilterPlot.png', bbox_inches="tight")
    plt.show()
