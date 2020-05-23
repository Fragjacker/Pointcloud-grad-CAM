import os

from matplotlib import pyplot as plt

import matplotlib as mpl
import numpy as np
from p_grad_CAM import maxNumPoints
from shared_functions.help_functions import getShapeName
import utils.test_data_handler as tdh

mpl.use('pdf')


def setNewPlot():
    width = 5.91
    height = width / 1.618
    plt.rc('font', family='serif', serif='Times')
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)
    plt.rc('legend', fontsize=6)  # using a size in points
    plt.rc('legend', title_fontsize=8)
    plt.figure(figsize=(width, height))


def autolabel(rects, plot):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = int(rect.get_height())
        plot.annotate('{}'.format(height),
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 1),  # 3 points vertical offset
                      textcoords="offset points",
                      fontsize=5,
                      ha='center', va='bottom')


def plotAveragePerformanceAsBars(CAMFilepath, SaliencyFilepath):
    fileDataCAM = [f for f in os.listdir(CAMFilepath) if os.path.isfile(os.path.join(CAMFilepath, f))]
    fileDataSaliency = [f for f in os.listdir(SaliencyFilepath) if os.path.isfile(os.path.join(SaliencyFilepath, f))]
    timePerformanceCAM = []
    timePerformanceSaliency = []
    bar_width = 1
    opacity = 0.6
    for eachFile in fileDataCAM:
        curPath = os.path.join(CAMFilepath, eachFile)
        curFile = tdh.readTestFile(curPath)
        timePerformanceCAM.append(curFile[0])
    for eachFile in fileDataSaliency:
        curPath = os.path.join(SaliencyFilepath, eachFile)
        curFile = tdh.readTestFile(curPath)
        timePerformanceSaliency.append(curFile[0])
    pgradCAMAvg = sum(timePerformanceCAM) / 40
    saliencyAvg = sum(timePerformanceSaliency) / 40

    width = 5.91
    height = width / 1.618

    plt.rc('font', family='serif', serif='Times')
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    plt.figure(figsize=(width, height))
    rects1 = plt.bar(1, pgradCAMAvg, width=bar_width, color='b', alpha=opacity, label="p-grad-CAM")
    rects2 = plt.bar(1 + bar_width, saliencyAvg, width=bar_width, color='g', alpha=opacity, label="ASM")

    plt.title("Average amount of time needed per object for each algorithm")
    plt.ylabel("Time in seconds")
    plt.xlabel("Point cloud objects")
    autolabel(rects1, plt)
    autolabel(rects2, plt)
    plt.xticks(np.arange(2) + bar_width, ('p-grad-CAM', 'ASM'))
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig('performance_average_time_comparison.pdf')
    plt.close()


def plotPerformanceAsBars(CAMFilepath, SaliencyFilepath):
    fileDataCAM = [f for f in os.listdir(CAMFilepath) if os.path.isfile(os.path.join(CAMFilepath, f))]
    fileDataSaliency = [f for f in os.listdir(SaliencyFilepath) if os.path.isfile(os.path.join(SaliencyFilepath, f))]
    bar_width = 0.45
    opacity = 0.6
    timePerformanceCAM = []
    timePerformanceSaliency = []
    newfileDataCAM = fileDataCAM[:round(len(fileDataCAM) / 2)]
    newfileDataSaliency = fileDataSaliency[:round(len(fileDataSaliency) / 2)]
    for eachFile in newfileDataCAM:
        curPath = os.path.join(CAMFilepath, eachFile)
        curFile = tdh.readTestFile(curPath)
        timePerformanceCAM.append(curFile[0])
    for eachFile in newfileDataSaliency:
        curPath = os.path.join(SaliencyFilepath, eachFile)
        curFile = tdh.readTestFile(curPath)
        timePerformanceSaliency.append(curFile[0])

    setNewPlot()

    rects1 = plt.bar(np.arange(len(newfileDataCAM)), timePerformanceCAM, width=bar_width, color='b', alpha=opacity,
                     label="p-grad-CAM")
    rects2 = plt.bar(np.arange(len(newfileDataSaliency)) + bar_width, timePerformanceSaliency, width=bar_width,
                     color='g',
                     alpha=opacity, label="ASM")

    plt.legend(title=("Used algorithms"))
    plt.title("Average amount of time needed per object for each algorithm")
    plt.ylabel("Time in seconds")
    plt.xlabel("Point cloud objects")
    autolabel(rects1, plt)
    autolabel(rects2, plt)
    plt.xticks(np.arange(20) + bar_width / 2, newfileDataCAM, rotation=45.0)
    plt.subplots_adjust(left=0.095, bottom=0.17, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig('performance_time_per_object_comparison_partI.pdf')
    plt.close()

    # ------------------------------------------------------------------------------

    timePerformanceCAM = []
    timePerformanceSaliency = []
    newfileDataCAM = fileDataCAM[round(len(fileDataCAM) / 2):]
    newfileDataSaliency = fileDataSaliency[round(len(fileDataSaliency) / 2):]
    for eachFile in newfileDataCAM:
        curPath = os.path.join(CAMFilepath, eachFile)
        curFile = tdh.readTestFile(curPath)
        timePerformanceCAM.append(curFile[0])
    for eachFile in newfileDataSaliency:
        curPath = os.path.join(SaliencyFilepath, eachFile)
        curFile = tdh.readTestFile(curPath)
        timePerformanceSaliency.append(curFile[0])

    setNewPlot()

    rects1 = plt.bar(np.arange(len(newfileDataCAM)), timePerformanceCAM, width=bar_width, color='b', alpha=opacity,
                     label="p-grad-CAM")
    rects2 = plt.bar(np.arange(len(newfileDataSaliency)) + bar_width, timePerformanceSaliency, width=bar_width,
                     color='g',
                     alpha=opacity, label="ASM")

    plt.legend(title=("Used algorithms"))
    plt.title("Average amount of time needed per object for each algorithm")
    plt.ylabel("Time in seconds")
    plt.xlabel("Point cloud objects")
    autolabel(rects1, plt)
    autolabel(rects2, plt)
    plt.xticks(np.arange(20) + bar_width / 2, newfileDataCAM, rotation=45.0)
    plt.subplots_adjust(left=0.095, bottom=0.17, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig('performance_time_per_object_comparison_partII.pdf')
    plt.close()


def plotAllFilesAverageAsBars(CAMFilepath, SaliencyFilepath):
    fileDataCAM = [f for f in os.listdir(CAMFilepath) if os.path.isfile(os.path.join(CAMFilepath, f))]
    fileDataSaliency = [f for f in os.listdir(SaliencyFilepath) if os.path.isfile(os.path.join(SaliencyFilepath, f))]
    numMaxRemovedPointsCAM = []
    numMaxRemovedPointsSaliency = []
    bar_width = 1
    opacity = 0.6
    for eachFile in fileDataCAM:
        curPath = os.path.join(CAMFilepath, eachFile)
        curFile = tdh.readTestFile(curPath)
        numMaxRemovedPointsCAM.append(maxNumPoints - curFile[len(curFile) - 1])
    for eachFile in fileDataSaliency:
        curPath = os.path.join(SaliencyFilepath, eachFile)
        curFile = tdh.readTestFile(curPath)
        numMaxRemovedPointsSaliency.append(maxNumPoints - curFile[len(curFile) - 1])
    pgradCAMAvg = sum(numMaxRemovedPointsCAM) / 40
    saliencyAvg = sum(numMaxRemovedPointsSaliency) / 40

    setNewPlot()

    rects1 = plt.bar(1, pgradCAMAvg, width=bar_width, color='b', alpha=opacity, label="p-grad-CAM")
    rects2 = plt.bar(1 + bar_width, saliencyAvg, width=bar_width, color='g', alpha=opacity, label="ASM")

    plt.title("Average amount of points removed for each algorithm")
    plt.ylabel("Points removed")
    autolabel(rects1, plt)
    autolabel(rects2, plt)
    plt.xticks(np.arange(2) + bar_width, ('p-grad-CAM', 'ASM'))
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig('performance_average_points_removed_comparison.pdf')
    plt.close()


def plotAllFilesAsBars(CAMFilepath, SaliencyFilepath):
    fileDataCAM = [f for f in os.listdir(CAMFilepath) if os.path.isfile(os.path.join(CAMFilepath, f))]
    fileDataSaliency = [f for f in os.listdir(SaliencyFilepath) if os.path.isfile(os.path.join(SaliencyFilepath, f))]
    numMaxRemovedPointsCAM = []
    numMaxRemovedPointsSaliency = []
    bar_width = 0.45
    opacity = 0.6
    newfileDataCAM = fileDataCAM[:round(len(fileDataCAM) / 2)]
    newfileDataSaliency = fileDataSaliency[:round(len(fileDataSaliency) / 2)]
    for eachFile in newfileDataCAM:
        curPath = os.path.join(CAMFilepath, eachFile)
        curFile = tdh.readTestFile(curPath)
        numMaxRemovedPointsCAM.append(maxNumPoints - curFile[len(curFile) - 1])
    for eachFile in newfileDataSaliency:
        curPath = os.path.join(SaliencyFilepath, eachFile)
        curFile = tdh.readTestFile(curPath)
        numMaxRemovedPointsSaliency.append(maxNumPoints - curFile[len(curFile) - 1])

    setNewPlot()

    rects1 = plt.bar(np.arange(len(newfileDataCAM)), numMaxRemovedPointsCAM, width=bar_width, color='b', alpha=opacity,
                     label="p-grad-CAM")
    rects2 = plt.bar(np.arange(len(newfileDataSaliency)) + bar_width, numMaxRemovedPointsSaliency, width=bar_width,
                     color='g', alpha=opacity, label="ASM")

    plt.legend(title=("Used algorithms"))
    plt.title("Total amount of points per object removed for each algorithm")
    plt.ylabel("Points removed")
    plt.xlabel("Point cloud objects")
    autolabel(rects1, plt)
    autolabel(rects2, plt)
    plt.xticks(np.arange(20) + bar_width / 2, newfileDataCAM, rotation=45.0)
    plt.subplots_adjust(left=0.095, bottom=0.17, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig('performance_points_removed_per_object_partI.pdf')
    plt.close()

    # ------------------------------------------------------------------------------
    numMaxRemovedPointsCAM = []
    numMaxRemovedPointsSaliency = []
    newfileDataCAM = fileDataCAM[round(len(fileDataCAM) / 2):]
    newfileDataSaliency = fileDataSaliency[round(len(fileDataSaliency) / 2):]
    for eachFile in newfileDataCAM:
        curPath = os.path.join(CAMFilepath, eachFile)
        curFile = tdh.readTestFile(curPath)
        numMaxRemovedPointsCAM.append(maxNumPoints - curFile[len(curFile) - 1])
    for eachFile in newfileDataSaliency:
        curPath = os.path.join(SaliencyFilepath, eachFile)
        curFile = tdh.readTestFile(curPath)
        numMaxRemovedPointsSaliency.append(maxNumPoints - curFile[len(curFile) - 1])

    setNewPlot()

    rects1 = plt.bar(np.arange(len(newfileDataCAM)), numMaxRemovedPointsCAM, width=bar_width, color='b', alpha=opacity,
                     label="p-grad-CAM")
    rects2 = plt.bar(np.arange(len(newfileDataSaliency)) + bar_width, numMaxRemovedPointsSaliency, width=bar_width,
                     color='g', alpha=opacity, label="ASM")

    plt.legend(title=("Used algorithms"))
    plt.title("Total amount of points per object removed for each algorithm")
    plt.ylabel("Points removed")
    plt.xlabel("Point cloud objects")
    autolabel(rects1, plt)
    autolabel(rects2, plt)
    plt.xticks(np.arange(20) + bar_width / 2, newfileDataCAM, rotation=45.0)
    plt.subplots_adjust(left=0.095, bottom=0.17, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig('performance_points_removed_per_object_partII.pdf')
    plt.close()


def plotAllFiles(CAMFilepath, SaliencyFilepath):
    start = 0
    end = 5
    counter = 0
    while end < 40:
        counter += 1
        fileDataCAM_points_removed = [f for f in os.listdir(CAMFilepath) if
                                      (os.path.isfile(os.path.join(CAMFilepath, f)) and '_points_removed' in f)]
        fileDataCAM_accuracy = [f for f in os.listdir(CAMFilepath) if
                                (os.path.isfile(os.path.join(CAMFilepath, f)) and '_accuracy' in f)]
        fileDataSaliency_points_removed = [f for f in os.listdir(SaliencyFilepath) if
                                           (os.path.isfile(
                                               os.path.join(SaliencyFilepath, f)) and '_points_removed' in f)]
        fileDataSaliency_accuracy = [f for f in os.listdir(SaliencyFilepath) if
                                     (os.path.isfile(os.path.join(SaliencyFilepath, f)) and '_accuracy' in f)]
        fileRange = range(start, end)

        setNewPlot()

        flag = False
        for index in fileRange:
            curAccPath = os.path.join(CAMFilepath, fileDataCAM_accuracy[index])
            curPrPath = os.path.join(CAMFilepath, fileDataCAM_points_removed[index])
            accuracyFile = tdh.readTestFile(curAccPath)
            prFile = tdh.readTestFile(curPrPath)
            if flag:
                plt.plot(prFile, accuracyFile, 'C0', label="p-grad-CAM")
                flag = False
            else:
                #             plt.plot( prFile, accuracyFile, 'C0')
                plt.plot(prFile, accuracyFile, label=getShapeName(index) + ": p-grad-CAM")
        #             plt.plot( prFile, accuracyFile )

        flag = False
        for index in fileRange:
            curAccPath = os.path.join(SaliencyFilepath, fileDataSaliency_accuracy[index])
            curPrPath = os.path.join(SaliencyFilepath, fileDataSaliency_points_removed[index])
            accuracyFile = tdh.readTestFile(curAccPath)
            prFile = tdh.readTestFile(curPrPath)
            if flag:
                plt.plot(prFile, accuracyFile, 'C1', label="saliency maps")
                flag = False
            else:
                #             plt.plot( prFile, accuracyFile, 'C1')
                plt.plot(prFile, accuracyFile, label=getShapeName(index) + ": ASM")

        plt.legend(title=("Shape and algorithm"))
        plt.title("Prediction accuracies per remaining points for shapes %s-%s" % (start, end))
        plt.ylabel("Accuracy")
        plt.xlabel("Remaining Points")
        plt.subplots_adjust(left=0.075, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
        plt.gca().invert_xaxis()
        plt.savefig('performance_points_removed_per_iteration_p-grad-CAM_part%s.pdf' % counter)
        plt.close()
        start += 5
        end += 5


def plotXYZRotatedResults():
    setNewPlot()
    savePath = os.path.join(os.path.split(__file__)[0], "result_data")

    airplaneAcc = tdh.readTestFile(os.path.join(savePath, "airplane_1000_0_Y_uncorrected_maxpooled_accuracy"))
    airplaneNonRotAcc = tdh.readTestFile(os.path.join(savePath, "airplane_1000_0_XYZ_uncorrected_maxpooled_accuracy"))
    plt.plot(np.arange(len(airplaneAcc)), airplaneAcc, label="Y rotation")
    plt.plot(np.arange(len(airplaneNonRotAcc)), airplaneNonRotAcc, label="XYZ rotation")
    plt.legend(title=("Rotation method"))
    plt.title("Original network handling different rotation methods")
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.subplots_adjust(left=0.085, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig('airplane_xyz_original_accuracy.pdf')
    plt.close()

    setNewPlot()
    airplaneAcc = tdh.readTestFile(os.path.join(savePath, "airplane_1000_0_Y_uncorrected_maxpooled_meanloss"))
    airplaneNonRotAcc = tdh.readTestFile(os.path.join(savePath, "airplane_1000_0_XYZ_uncorrected_maxpooled_meanloss"))
    plt.plot(np.arange(len(airplaneAcc)), airplaneAcc, label="Y rotation")
    plt.plot(np.arange(len(airplaneNonRotAcc)), airplaneNonRotAcc, label="XYZ rotation")
    plt.legend(title=("Rotation method"))
    plt.title("Original network handling different rotation methods")
    plt.ylabel("Meanloss")
    plt.xlabel("Iterations")
    plt.subplots_adjust(left=0.085, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig('airplane_xyz_original_meanloss.pdf')
    plt.close()

    # ------------------------------------------------------------------------------

    setNewPlot()
    airplaneAcc = tdh.readTestFile(os.path.join(savePath, "airplane_1000_0_Y_corrected_maxpooled_accuracy"))
    airplaneNonRotAcc = tdh.readTestFile(os.path.join(savePath, "airplane_1000_0_XYZ_corrected_maxpooled_accuracy"))
    plt.plot(np.arange(len(airplaneAcc)), airplaneAcc, label="Y rotation")
    plt.plot(np.arange(len(airplaneNonRotAcc)), airplaneNonRotAcc, label="XYZ rotation")
    plt.legend(title=("Rotation method"))
    plt.title("Retrained fixed network handling different rotation methods")
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.subplots_adjust(left=0.085, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig('airplane_xyz_retrained_accuracy.pdf')
    plt.close()

    setNewPlot()
    airplaneAcc = tdh.readTestFile(os.path.join(savePath, "airplane_1000_0_Y_corrected_maxpooled_meanloss"))
    airplaneNonRotAcc = tdh.readTestFile(os.path.join(savePath, "airplane_1000_0_XYZ_corrected_maxpooled_meanloss"))
    plt.plot(np.arange(len(airplaneAcc)), airplaneAcc, label="Y rotation")
    plt.plot(np.arange(len(airplaneNonRotAcc)), airplaneNonRotAcc, label="XYZ rotation")
    plt.legend(title=("Rotation method"))
    plt.title("Retrained fixed network handling different rotation methods")
    plt.ylabel("Meanloss")
    plt.xlabel("Iterations")
    plt.subplots_adjust(left=0.085, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig('airplane_xyz_retrained_meanloss.pdf')
    plt.close()


def plotAlgoComparison():
    # Airplane
    setNewPlot()
    curShape = getShapeName(0)
    savePath = os.path.join(os.path.split(__file__)[0], "testdata")
    pGradCAMpointsRemovedPlot = tdh.readTestFile(os.path.join(savePath, curShape + "_p-grad-CAM_removed"))
    saliencyRemovedPlot = tdh.readTestFile(os.path.join(savePath, curShape + "_saliency_removed"))
    plt.plot(np.arange(len(pGradCAMpointsRemovedPlot)), pGradCAMpointsRemovedPlot, label="p-grad-CAM")
    plt.plot(np.arange(len(saliencyRemovedPlot)), saliencyRemovedPlot, label="saliency maps")
    plt.legend(title=("Used algorithms"))
    plt.title(curShape + " remaining points per iteration")
    plt.ylabel("Points")
    plt.xlabel("Iterations")
    plt.subplots_adjust(left=0.095, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig(curShape + '_point_removal_comparison.pdf')
    plt.close()
    # Car
    setNewPlot()
    curShape = getShapeName(7)
    savePath = os.path.join(os.path.split(__file__)[0], "testdata")
    pGradCAMpointsRemovedPlot = tdh.readTestFile(os.path.join(savePath, curShape + "_p-grad-CAM_removed"))
    saliencyRemovedPlot = tdh.readTestFile(os.path.join(savePath, curShape + "_saliency_removed"))
    plt.plot(np.arange(len(pGradCAMpointsRemovedPlot)), pGradCAMpointsRemovedPlot, label="p-grad-CAM")
    plt.plot(np.arange(len(saliencyRemovedPlot)), saliencyRemovedPlot, label="saliency maps")
    plt.legend(title=("Used algorithms"))
    plt.title(curShape + " remaining points per iteration")
    plt.ylabel("Points")
    plt.xlabel("Iterations")
    plt.subplots_adjust(left=0.095, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig(curShape + '_point_removal_comparison.pdf')
    plt.close()
    # Cone
    setNewPlot()
    curShape = getShapeName(9)
    savePath = os.path.join(os.path.split(__file__)[0], "testdata")
    pGradCAMpointsRemovedPlot = tdh.readTestFile(os.path.join(savePath, curShape + "_p-grad-CAM_removed"))
    saliencyRemovedPlot = tdh.readTestFile(os.path.join(savePath, curShape + "_saliency_removed"))
    plt.plot(np.arange(len(pGradCAMpointsRemovedPlot)), pGradCAMpointsRemovedPlot, label="p-grad-CAM")
    plt.plot(np.arange(len(saliencyRemovedPlot)), saliencyRemovedPlot, label="saliency maps")
    plt.legend(title=("Used algorithms"))
    plt.title(curShape + " remaining points per iteration")
    plt.ylabel("Points")
    plt.xlabel("Iterations")
    plt.subplots_adjust(left=0.095, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig(curShape + '_point_removal_comparison.pdf')
    plt.close()
    # Guitar
    setNewPlot()
    curShape = getShapeName(17)
    savePath = os.path.join(os.path.split(__file__)[0], "testdata")
    pGradCAMpointsRemovedPlot = tdh.readTestFile(os.path.join(savePath, curShape + "_p-grad-CAM_removed"))
    saliencyRemovedPlot = tdh.readTestFile(os.path.join(savePath, curShape + "_saliency_removed"))
    plt.plot(np.arange(len(pGradCAMpointsRemovedPlot)), pGradCAMpointsRemovedPlot, label="p-grad-CAM")
    plt.plot(np.arange(len(saliencyRemovedPlot)), saliencyRemovedPlot, label="saliency maps")
    plt.legend(title=("Used algorithms"))
    plt.title(curShape + " remaining points per iteration")
    plt.ylabel("Points")
    plt.xlabel("Iterations")
    plt.subplots_adjust(left=0.095, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig(curShape + '_point_removal_comparison.pdf')
    plt.close()


def createDir(dirName):
    targetDir = os.path.join(os.path.split(__file__)[0], dirName)
    if not os.path.exists(targetDir):
        os.mkdir(targetDir)
    return targetDir


def plotAllPPIAccuracy():
    for eachShape in range(40):
        dumpDir = createDir("PPI_accuracy_plots")
        setNewPlot()
        curShape = getShapeName(eachShape)
        savePath = os.path.join(os.path.split(__file__)[0], "testdata")
        pGradCAMpointsRemovedPlot = tdh.readTestFile(
            os.path.join(savePath, "p-grad-CAM_ppi", curShape + "_points_removed"))
        pGradCAMpointsAccuracyPlot = tdh.readTestFile(os.path.join(savePath, "p-grad-CAM_ppi", curShape + "_accuracy"))
        saliencyRemovedPlot = tdh.readTestFile(
            os.path.join(savePath, "saliency_maps_ppi", curShape + "_points_removed"))
        saliencyAccuracyPlot = tdh.readTestFile(os.path.join(savePath, "saliency_maps_ppi", curShape + "_accuracy"))
        plt.plot(pGradCAMpointsRemovedPlot, pGradCAMpointsAccuracyPlot, label="p-grad-CAM Points")
        plt.plot(saliencyRemovedPlot, saliencyAccuracyPlot, label="saliency maps Points")
        plt.legend(title=("Used algorithms"))
        plt.title(curShape + " accuracy per remaining points per iteration")
        plt.ylabel("Accuracy")
        plt.xlabel("Remaining Points")
        plt.gca().invert_xaxis()
        plt.subplots_adjust(left=0.075, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
        plt.savefig(os.path.join(dumpDir, curShape + '_point_removal_accuracy_comparison.pdf'))
        plt.close()


def plotThresholdTest(curShape):
    dumpDir = createDir("Threshold_test_plots")
    numTestRuns = 500
    # ===========================================================================
    # Max pooling
    # ===========================================================================
    savePath = os.path.join(os.path.split(__file__)[0], "testdata", curShape)
    testResultsacc0 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+average_accuracy"))
    testResultsacc1 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+median_accuracy"))
    testResultsacc2 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+midrange_accuracy"))
    testResultsacc6 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+random_accuracy"))
    testResultsacc7 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_nonzero_accuracy"))
    testResultsloss0 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+average_meanloss"))
    testResultsloss1 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+median_meanloss"))
    testResultsloss2 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+midrange_meanloss"))
    testResultsloss6 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_+random_meanloss"))
    testResultsloss7 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_nonzero_meanloss"))
    vuptestResultsacc0 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-average_accuracy"))
    vuptestResultsacc1 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-median_accuracy"))
    vuptestResultsacc2 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-midrange_accuracy"))
    vuptestResultsacc6 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-random_accuracy"))
    vuptestResultsacc7 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_zero_accuracy"))
    vuptestResultsloss0 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-average_meanloss"))
    vuptestResultsloss1 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-median_meanloss"))
    vuptestResultsloss2 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-midrange_meanloss"))
    vuptestResultsloss6 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_-random_meanloss"))
    vuptestResultsloss7 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_maxpooling_zero_meanloss"))
    # ===============================================================================
    # Average pooling
    # ===============================================================================
    avgtestResultsacc0 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+average_accuracy"))
    avgtestResultsacc1 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+median_accuracy"))
    avgtestResultsacc2 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+midrange_accuracy"))
    avgtestResultsacc6 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+random_accuracy"))
    avgtestResultsacc7 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_nonzero_accuracy"))
    avgtestResultsloss0 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+average_meanloss"))
    avgtestResultsloss1 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+median_meanloss"))
    avgtestResultsloss2 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+midrange_meanloss"))
    avgtestResultsloss6 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_+random_meanloss"))
    avgtestResultsloss7 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_nonzero_meanloss"))
    vupavgtestResultsacc0 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-average_accuracy"))
    vupavgtestResultsacc1 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-median_accuracy"))
    vupavgtestResultsacc2 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-midrange_accuracy"))
    vupavgtestResultsacc6 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-random_accuracy"))
    vupavgtestResultsacc7 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_zero_accuracy"))
    vupavgtestResultsloss0 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-average_meanloss"))
    vupavgtestResultsloss1 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-median_meanloss"))
    vupavgtestResultsloss2 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-midrange_meanloss"))
    vupavgtestResultsloss6 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_-random_meanloss"))
    vupavgtestResultsloss7 = tdh.readTestFile(
        os.path.join(savePath, curShape + "_" + str(numTestRuns) + "_XYZ_avgpooling_zero_meanloss"))
    # ===============================================================================
    # Plot the results now
    # ===============================================================================
    setNewPlot()
    plt.plot(np.arange(len(testResultsacc0)), testResultsacc0, label="Maxpooled average removed")
    plt.plot(np.arange(len(testResultsacc1)), testResultsacc1, label="Maxpooled median removed")
    plt.plot(np.arange(len(testResultsacc2)), testResultsacc2, label="Maxpooled midrange removed")
    plt.plot(np.arange(len(testResultsacc6)), testResultsacc6, label="Maxpooled Random removed")
    plt.plot(np.arange(len(testResultsacc7)), testResultsacc7, label="Maxpooled Non Zeros removed")
    plt.plot(np.arange(len(avgtestResultsacc0)), avgtestResultsacc0, label="Average pooled average removed")
    plt.plot(np.arange(len(avgtestResultsacc1)), avgtestResultsacc1, label="Average pooled median removed")
    plt.plot(np.arange(len(avgtestResultsacc2)), avgtestResultsacc2, label="Average pooled midrange removed")
    plt.plot(np.arange(len(avgtestResultsacc2)), avgtestResultsacc6, label="Average pooled Random removed")
    plt.plot(np.arange(len(avgtestResultsacc7)), avgtestResultsacc7, label="Average pooled Non Zeros removed")
    plt.legend(title=(curShape + " important point removal plot"))
    plt.title(curShape + " removing important points accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig(os.path.join(dumpDir, 'vip_' + curShape + '_accuracy.pdf'))
    plt.close()
    setNewPlot()
    plt.plot(np.arange(len(vuptestResultsacc0)), vuptestResultsacc0, label="Maxpooled average removed")
    plt.plot(np.arange(len(vuptestResultsacc1)), vuptestResultsacc1, label="Maxpooled median removed")
    plt.plot(np.arange(len(vuptestResultsacc2)), vuptestResultsacc2, label="Maxpooled midrange removed")
    plt.plot(np.arange(len(vuptestResultsacc6)), vuptestResultsacc6, label="Maxpooled Random removed")
    plt.plot(np.arange(len(vuptestResultsacc7)), vuptestResultsacc7, label="Maxpooled Zeros removed")
    plt.plot(np.arange(len(vupavgtestResultsacc0)), vupavgtestResultsacc0, label="Average pooled average removed")
    plt.plot(np.arange(len(vupavgtestResultsacc1)), vupavgtestResultsacc1, label="Average pooled median removed")
    plt.plot(np.arange(len(vupavgtestResultsacc2)), vupavgtestResultsacc2, label="Average pooled midrange removed")
    plt.plot(np.arange(len(vupavgtestResultsacc6)), vupavgtestResultsacc6, label="Average pooled Random removed")
    plt.plot(np.arange(len(vupavgtestResultsacc7)), vupavgtestResultsacc7, label="Average pooled Zeros removed")
    plt.legend(title=(curShape + " unimportant point removal plot"))
    plt.title(curShape + " removing unimportant points accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig(os.path.join(dumpDir, 'vup_' + curShape + '_accuracy.pdf'))
    plt.close()
    setNewPlot()
    plt.plot(np.arange(len(testResultsloss0)), testResultsloss0, label="Maxpooled average removed")
    plt.plot(np.arange(len(testResultsloss1)), testResultsloss1, label="Maxpooled median removed")
    plt.plot(np.arange(len(testResultsloss2)), testResultsloss2, label="Maxpooled midrange removed")
    plt.plot(np.arange(len(testResultsloss6)), testResultsloss6, label="Maxpooled Random removed")
    plt.plot(np.arange(len(testResultsloss7)), testResultsloss7, label="Maxpooled Non Zeros removed")
    plt.plot(np.arange(len(avgtestResultsloss0)), avgtestResultsloss0, label="Average pooled average removed")
    plt.plot(np.arange(len(avgtestResultsloss1)), avgtestResultsloss1, label="Average pooled median removed")
    plt.plot(np.arange(len(avgtestResultsloss2)), avgtestResultsloss2, label="Average pooled midrange removed")
    plt.plot(np.arange(len(avgtestResultsloss6)), avgtestResultsloss6, label="Average pooled Random removed")
    plt.plot(np.arange(len(avgtestResultsloss7)), avgtestResultsloss7, label="Average pooled Non Zeros removed")
    plt.legend(title=(curShape + " important point removal plot"))
    plt.title(curShape + " removing important points loss")
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.subplots_adjust(left=0.07, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig(os.path.join(dumpDir, 'vip_' + curShape + '_meanloss.pdf'))
    plt.close()
    setNewPlot()
    plt.plot(np.arange(len(vuptestResultsloss0)), vuptestResultsloss0, label="Maxpooled average removed")
    plt.plot(np.arange(len(vuptestResultsloss1)), vuptestResultsloss1, label="Maxpooled median removed")
    plt.plot(np.arange(len(vuptestResultsloss2)), vuptestResultsloss2, label="Maxpooled midrange removed")
    plt.plot(np.arange(len(vuptestResultsloss6)), vuptestResultsloss6, label="Maxpooled Random removed")
    plt.plot(np.arange(len(vuptestResultsloss7)), vuptestResultsloss7, label="Maxpooled Zeros removed")
    plt.plot(np.arange(len(vupavgtestResultsloss0)), vupavgtestResultsloss0, label="Average pooled average removed")
    plt.plot(np.arange(len(vupavgtestResultsloss1)), vupavgtestResultsloss1, label="Average pooled median removed")
    plt.plot(np.arange(len(vupavgtestResultsloss2)), vupavgtestResultsloss2, label="Average pooled midrange removed")
    plt.plot(np.arange(len(vupavgtestResultsloss6)), vupavgtestResultsloss6, label="Average pooled Random removed")
    plt.plot(np.arange(len(vupavgtestResultsloss7)), vupavgtestResultsloss7, label="Average pooled Zeros removed")
    plt.legend(title=(curShape + " unimportant point removal plot"))
    plt.title(curShape + " removing unimportant points loss")
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.subplots_adjust(left=0.07, bottom=0.1, right=0.999, top=0.935, wspace=0.2, hspace=0.17)
    plt.savefig(os.path.join(dumpDir, 'vup_' + curShape + '_meanloss.pdf'))
    plt.close()


# ===============================================================================
# EXECUTE PLOTS
# ===============================================================================
plotAllFiles(os.path.join("testdata", "p-grad-CAM_ppi"), os.path.join("testdata", "saliency_maps_ppi"))
plotAllFilesAsBars(os.path.join("testdata", "p-grad-CAM"), os.path.join("testdata", "saliency_maps"))
plotAllFilesAverageAsBars(os.path.join("testdata", "p-grad-CAM"), os.path.join("testdata", "saliency_maps"))
plotPerformanceAsBars(os.path.join("testdata", "p-grad-CAM_performance"),
                      os.path.join("testdata", "saliency_maps_performance"))
plotAveragePerformanceAsBars(os.path.join("testdata", "p-grad-CAM_performance"),
                             os.path.join("testdata", "saliency_maps_performance"))
plotXYZRotatedResults()
plotAlgoComparison()
plotAllPPIAccuracy()

plotThresholdTest(getShapeName(0))  # Airplane
plotThresholdTest(getShapeName(1))  # Bathtub
plotThresholdTest(getShapeName(2))  # Bed
plotThresholdTest(getShapeName(3))  # Bench
plotThresholdTest(getShapeName(4))  # Bookshelf
plotThresholdTest(getShapeName(7))  # Car
plotThresholdTest(getShapeName(9))  # Cone
plotThresholdTest(getShapeName(24))  # Person
