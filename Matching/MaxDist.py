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
    minHamming_prefilters = [False, 20, 50, 70]
    maxHamming_postfilters = range(0, 256, 5)
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

    with open("output/MaxDist.txt", 'w', encoding='utf-8') as f:
        f.write(tabulate(data, headers=headers))

    data2 = pd.DataFrame(data, columns=headers)

    x = maxHamming_postfilters
    y_PreFilterFalse = [data2[data2['Prefilt'] == '-'].groupby('Max.Dist.')['#Correct'].get_group(i).values[0] for i in maxHamming_postfilters]
    y_PreFilter20 = [data2[data2['Prefilt'] == minHamming_prefilters[1]].groupby('Max.Dist.')['#Correct'].get_group(i).values[0] for i in maxHamming_postfilters]
    y_PreFilter50 = [data2[data2['Prefilt'] == minHamming_prefilters[2]].groupby('Max.Dist.')['#Correct'].get_group(i).values[0] for i in maxHamming_postfilters]
    y_PreFilter70 = [data2[data2['Prefilt'] == minHamming_prefilters[3]].groupby('Max.Dist.')['#Correct'].get_group(i).values[0] for i in maxHamming_postfilters]

    line_PreFilterFalse = plt.plot(x, y_PreFilterFalse, 'g-', x, y_PreFilterFalse, 'g^')
    line_PreFilter20 = plt.plot(x, y_PreFilter20, 'b-', x, y_PreFilter20, 'b+')
    line_PreFilter50 = plt.plot(x, y_PreFilter50, 'r-', x, y_PreFilter50, 'rx')
    line_PreFilter70 = plt.plot(x, y_PreFilter70, 'y-', x, y_PreFilter70, 'yd')

    plt.title('Number of correct match')
    plt.xlabel('Distance value')
    plt.ylabel('#Correct')

    legends = ['Prefilter: ' + str(i) for i in minHamming_prefilters]

    plt.legend([line_PreFilterFalse[0], line_PreFilter20[0], line_PreFilter50[0], line_PreFilter70[0]], legends)

    plt.savefig('output/#MaxDist.png', bbox_inches="tight")
    plt.show()

    # ---------------------
    y_PreFilterFalse = [data2[data2['Prefilt'] == '-'].groupby('Max.Dist.')['%Correct'].get_group(i).values[0] for i in maxHamming_postfilters]
    y_PreFilter20 = [data2[data2['Prefilt'] == minHamming_prefilters[1]].groupby('Max.Dist.')['%Correct'].get_group(i).values[0] for i in maxHamming_postfilters]
    y_PreFilter50 = [data2[data2['Prefilt'] == minHamming_prefilters[2]].groupby('Max.Dist.')['%Correct'].get_group(i).values[0] for i in maxHamming_postfilters]
    y_PreFilter70 = [data2[data2['Prefilt'] == minHamming_prefilters[3]].groupby('Max.Dist.')['%Correct'].get_group(i).values[0] for i in maxHamming_postfilters]

    line_PreFilterFalse = plt.plot(x, y_PreFilterFalse, 'g-', x, y_PreFilterFalse, 'g^')
    line_PreFilter20 = plt.plot(x, y_PreFilter20, 'b-', x, y_PreFilter20, 'b+')
    line_PreFilter50 = plt.plot(x, y_PreFilter50, 'r-', x, y_PreFilter50, 'rx')
    line_PreFilter70 = plt.plot(x, y_PreFilter70, 'y-', x, y_PreFilter70, 'yd')

    plt.title('Percent of correct match')
    plt.xlabel('Distance value')
    plt.ylabel('%Correct')
    plt.legend([line_PreFilterFalse[0], line_PreFilter20[0], line_PreFilter50[0], line_PreFilter70[0]], legends)

    plt.savefig('output/%MaxDist.png', bbox_inches="tight")
    plt.show()
