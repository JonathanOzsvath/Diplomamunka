import cv2
import numpy as np
import matplotlib.pyplot as plt
import image_matcher_eval as ime
from tabulate import tabulate

if __name__ == '__main__':
    data = []
    summary = []
    methodId = 1
    name_ref = "darts1_1"

    name_perspectives = ['darts2_1', 'darts_alul', 'darts_bal', 'darts_felul', 'darts_jobb']
    minHamming_prefilters = [0, 20, 32, 70]
    # minHamming_prefilters = [0]
    maxHamming_postfilters = [35, 40, 255]
    # maxHamming_postfilters = [40]
    cross_Checks = [True, False]
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

    with open("output/CrossCheck.txt", 'w', encoding='utf-8') as f:
        f.write(tabulate(data, headers=headers))

    cross_Check_False = list(filter(lambda d: not d[9], data))
    cross_Check_True = list(filter(lambda d: d[9], data))

    x1 = [i[8] for i in cross_Check_True]
    # x1 = [a[:[i for i, n in enumerate(a) if n == '_'][1]] + '\n' + a[[i for i, n in enumerate(a) if n == '_'][1]:]for a in x1]
    y1 = [i[14] for i in cross_Check_True]
    y2 = [i[14] for i in cross_Check_False]

    N = len(x1)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    p1 = ax.bar(ind, y1, width, bottom=0)
    p2 = ax.bar(ind + width, y2, width, bottom=0)

    ax.set_title('Number of correct match')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(x1, rotation=45, horizontalalignment="right")

    ax.legend((p1[0], p2[0]), ('CrossCheck: True', 'CrossCheck: False'))
    ax.autoscale_view()

    plt.savefig('output/#CrossCheck.png', bbox_inches="tight")
    plt.show()

    # ---------------------
    y1 = [i[15] for i in cross_Check_True]
    y2 = [i[15] for i in cross_Check_False]

    fig, ax = plt.subplots()

    p1 = ax.bar(ind, y1, width, bottom=0)
    p2 = ax.bar(ind + width, y2, width, bottom=0)

    ax.set_title('Percent of correct match')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(x1, rotation=45, horizontalalignment="right")

    ax.legend((p1[0], p2[0]), ('CrossCheck: True', 'CrossCheck: False'))
    ax.autoscale_view()

    plt.savefig('output/%CrossCheck.png', bbox_inches="tight")
    plt.show()
