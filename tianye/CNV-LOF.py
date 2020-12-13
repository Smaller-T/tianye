
'''极坐标实现'''
import numpy as np
import sys
import pysam
import os
from scipy.stats import norm
import gc
import pandas as pd
import scipy
from numba import njit
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from sklearn import preprocessing
import datetime
from sklearn.cluster import KMeans

from sklearn.metrics import euclidean_distances


def get_chrlist(filename):
    samfile = pysam.AlignmentFile(filename, "rb")
    List = samfile.references
    chrList = np.full(len(List), 0)
    for i in range(len(List)):
        chr = str(List[i]).strip('chr')
        if chr.isdigit():
            chrList[i] = int(chr)

    index = chrList > 0
    chrList = chrList[index]
    return chrList	#返回所有碱基比对上references的染色体号 组成的列表


def get_RC(filename, chrList, ReadCount):
    samfile = pysam.AlignmentFile(filename, "rb")
    for line in samfile:
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit():
                num = np.argwhere(chrList == int(chr))[0][0]
                posList = line.positions
                ReadCount[num][posList] += 1
    return ReadCount


def read_ref_file(filename, ref, num):
    # read reference file
    if os.path.exists(filename):
        print("Read reference file: " + str(filename))
        with open(filename, 'r') as f:
            line = f.readline()
            for line in f:
                linestr = line.strip()
                ref[num] += linestr
    else:
        print("Warning: can not open " + str(filename) + '\n')
    return ref


def ReadDepth(ReadCount, binNum, ref):
    RD = np.full(binNum, 0.0)
    GC = np.full(binNum, 0)
    pos = np.arange(1, binNum+1)
    for i in range(binNum):
        RD[i] = np.mean(ReadCount[i*binSize:(i+1)*binSize])
        cur_ref = ref[i*binSize:(i+1)*binSize]
        N_count = cur_ref.count('N') + cur_ref.count('n')
        if N_count == 0:
            gc_count = cur_ref.count('C') + cur_ref.count('c') + cur_ref.count('G') + cur_ref.count('g')
        else:
            RD[i] = -10000
            gc_count = 0
        GC[i] = int(round(gc_count / binSize, 3) * 1000)

    index = RD >= 0
    RD = RD[index]
    GC = GC[index]
    pos = pos[index]
    RD = gc_correct(RD, GC)

    return pos, RD


def gc_correct(RD, GC):
    # correcting gc bias
    bincount = np.bincount(GC)
    global_rd_ave = np.mean(RD)
    for i in range(len(RD)):
        if bincount[GC[i]] < 2:
            continue
        mean = np.mean(RD[GC == GC[i]])
        RD[i] = global_rd_ave * RD[i] / mean
    return RD

def dis_matrix(RD_count):
    # calculating euclidean_distances matrix
    RD_count = RD_count.astype(np.float)
    pos = np.array(range(1, len(RD_count)+1))
    nr_min = np.min(RD_count)
    nr_max = np.max(RD_count)
    #newpos = (pos - min(pos)) / (max(pos) - min(pos)) * (nr_max - nr_min) + nr_min
    newpos = (pos) / (max(pos)) * 2 * np.pi	#newpos为pos转化为0-2pi之间
    RD_count = (RD_count)/(max(RD_count))#r要以1为基准
    RD_count = RD_count.astype(np.float)
    newpos = newpos.astype(np.float)
    rd = np.c_[newpos, RD_count]
    plot_polar(newpos,RD_count)
#################################将rd的两列变成r*cos theta
    temp = rd.copy()
    rd[:,0] = temp[:,1]*np.cos(temp[:,0])
    rd[:,1] = temp[:,1]*np.sin(temp[:,0])

    dis = euclidean_distances(rd, rd)

    return dis, newpos

@njit
def k_matrix(dis, k):
    min_matrix = np.zeros((dis.shape[0], k))
    for i in range(dis.shape[0]):
        sort = np.argsort(dis[i])
        min_row = dis[i][sort[k]]#####
        for j in range(1, k + 1):
            min_matrix[i, j] = sort[j]
        dis[i][sort[1:(k + 1)]] = min_row
    return dis, min_matrix


@njit
def reach_density(dis, min_matrix, k):
    density = []
    for i in range(min_matrix.shape[0]):
        cur_sum = np.sum(dis[min_matrix[i], i])
        if cur_sum == 0.0:
            cur_density = 100
        else:
            cur_density = 1 / (cur_sum / k)
        density.append(cur_density)
    return density


def get_scores(density, min_matrix, binHead, k):
    scores = np.full(int(len(binHead)), 0.0)
    for i in range(min_matrix.shape[0]):
        cur_rito = density[min_matrix[i]] / density[i]
        cur_sum = np.sum(cur_rito) / k
        scores[i] = cur_sum

    return scores


def scaling_RD(RD, mode):
    posiRD = RD[RD > mode]
    negeRD = RD[RD < mode]
    if len(posiRD) < 50:
        mean_max_RD = np.mean(posiRD)
    else:
        sort = np.argsort(posiRD)
        maxRD = posiRD[sort[-50:]]
        mean_max_RD = np.mean(maxRD)

    if len(negeRD) < 50:
        mean_min_RD = np.mean(negeRD)
    else:
        sort = np.argsort(negeRD)
        minRD = negeRD[sort[:50]]
        mean_min_RD = np.mean(minRD)
    scaling = mean_max_RD / (mode + mode - mean_min_RD)
    for i in range(len(RD)):
        if RD[i] < mode:
            RD[i] /= scaling
    return RD


def modeRD(RD):
    newRD = np.full(len(RD), 0)
    for i in range(len(RD)):
        #print(RD[i])
        newRD[i] = int(round(RD[i], 3) * 1000)

    count = np.bincount(newRD)
    countList = np.full(len(count) - 49, 0)
    for i in range(len(countList)):
        countList[i] = np.mean(count[i:i + 50])
    modemin = np.argmax(countList)
    modemax = modemin + 50
    mode = (modemax + modemin) / 2
    mode = mode / 1000
    return mode

def plot_polar(_theta,_r):	#scatter函数的参数是先theta后r
    ax = plt.subplot(111,projection='polar')
    c = ax.scatter(_theta,_r,c='b',cmap='hsv',alpha=0.8)
    plt.show()

def plot(pos, data):
    plt.scatter(pos, data, s=5, c="black")
    plt.xlabel("pos")
    plt.ylabel("RD")
    plt.show()


def seg_RD(RD, binHead, seg_start, seg_end, seg_count):
    seg_RD = np.full(len(seg_count), 0.0)
    for i in range(len(seg_RD)):
        seg_RD[i] = np.mean(RD[seg_start[i]:seg_end[i]])
        seg_start[i] = binHead[seg_start[i]] * binSize + 1
        if seg_end[i] == len(binHead):
            seg_end[i] = len(binHead) - 1
        seg_end[i] = binHead[seg_end[i]] * binSize + binSize

    return seg_RD, seg_start, seg_end


def Write_data_file(chr, seg_start, seg_end, seg_count, scores):
    """
    write data file
    pos_start, pos_end, lof_score, p_value
    """
    output = open(p_value_file, "w")
    output.write("start" + '\t' + "end" + '\t' + "read depth" + '\t' + "lof score" + '\t' + "p value" + '\n')
    for i in range(len(scores)):
        output.write(
            str(chr[i]) + '\t' + str(seg_start[i]) + '\t' + str(seg_end[i]) +
            '\t' + str(seg_count[i]) + '\t' + str(scores[i]) + '\n')


def Write_CNV_File(chr, CNVstart, CNVend, CNVtype, CN, filename):
    """
    write cnv result file
    pos start, pos end, type, copy number
    """
    output = open(filename, "w")
    for i in range(len(CNVtype)):
        if CNVtype[i] == 2:
            output.write("chr" + str(chr[i]) + '\t' + str(CNVstart[i]) + '\t' + str(
                CNVend[i]) + '\t' + str("gain") + '\t' + str(CN[i]) + '\n')
        else:
            output.write("chr" + str(chr[i]) + '\t' + str(CNVstart[i]) + '\t' + str(
                CNVend[i]) + '\t' + str("loss") + '\t' + str(CN[i]) + '\n')


def Read_seg_file(num_col, num_bin):
    """
    read segment file (Generated by DNAcopy.segment)
    seg file: col, chr, start, end, num_mark, seg_mean
    """
    seg_start = []
    seg_end = []
    seg_count = []
    seg_len = []
    with open("seg", 'r') as f:
        for line in f:
            linestrlist = line.strip().split('\t')
            start = (int(linestrlist[0]) - 1) * num_col + int(linestrlist[2]) - 1
            end = (int(linestrlist[0]) - 1) * num_col + int(linestrlist[3]) - 1
            if start < num_bin:
                if end > num_bin:
                    end = num_bin - 1
                seg_start.append(start)
                seg_end.append(end)
                seg_count.append(float(linestrlist[5]))
                seg_len.append(int(linestrlist[4]))
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)

    return seg_start, seg_end, seg_count, seg_len


def calculating_CN(mode, CNVRD, CNVtype):

    CN = np.full(len(CNVtype), 0)
    index = CNVtype == 1
    lossRD = CNVRD[index]
    if len(lossRD) > 2:
        data = np.c_[lossRD, lossRD]
        del_type = KMeans(n_clusters=2, random_state=9).fit_predict(data)
        CNVtype[index] = del_type
        if np.mean(lossRD[del_type == 0]) < np.mean(lossRD[del_type == 1]):
            homoRD = np.mean(lossRD[del_type == 0])
            hemiRD = np.mean(lossRD[del_type == 1])
            for i in range(len(CN)):
                if CNVtype[i] == 0:
                    CN[i] = 0
                elif CNVtype[i] == 1:
                    CN[i] = 1
        else:
            hemiRD = np.mean(lossRD[del_type == 0])
            homoRD = np.mean(lossRD[del_type == 1])
            for i in range(len(CN)):
                if CNVtype[i] == 1:
                    CN[i] = 0
                elif CNVtype[i] == 0:
                    CN[i] = 1
        purity = 2 * (homoRD - hemiRD) / (homoRD - 2 * hemiRD)

    else:
        purity = 0.5
    for i in range(len(CNVtype)):
        if CNVtype[i] == 2:
            CN[i] = int(2 * CNVRD[i] / (mode * purity) - 2 * (1 - purity) / purity)
    return CN


def boxplot(scores):
    four = pd.Series(scores).describe()
    Q1 = four['25%']
    Q3 = four['75%']
    IQR = Q3 - Q1
    upper = Q3 + 0.75 * IQR
    lower = Q1 - 0.75 * IQR
    return upper


def write(data, data1):
    output = open("rd.txt", "w")
    for i in range(len(data)):
        output.write(str(data[i]) + '\t' + str(data1[i]) + '\n')


def combiningCNV(seg_chr, seg_start, seg_end, seg_count, scores, upper, mode):

    index = scores > upper
    CNV_chr = seg_chr[index]
    CNVstart = seg_start[index]
    CNVend = seg_end[index]
    CNVRD = seg_count[index]

    type = np.full(len(CNVRD), 1)
    for i in range(len(CNVRD)):
        if CNVRD[i] > mode:
            type[i] = 2

    for i in range(len(CNVRD) - 1):
        if CNVend[i] + 1 == CNVstart[i + 1] and type[i] == type[i + 1]:
            CNVstart[i + 1] = CNVstart[i]
            type[i] = 0

    index = type != 0
    CNVRD = CNVRD[index]
    CNV_chr = CNV_chr[index]
    CNVstart = CNVstart[index]
    CNVend = CNVend[index]
    CNVtype = type[index]

    return CNV_chr, CNVstart, CNVend, CNVRD, CNVtype


# get params

starttime = datetime.datetime.now()
bam = sys.argv[1]
binSize = 1000
path = os.path.abspath('.')
segpath = path+str("/seg")
p_value_file = bam + ".score.txt"
outfile = bam + "_result.txt"


all_chr = []
all_RD = []
all_start = []
all_end = []
chrList = get_chrlist(bam)
chrNum = len(chrList)
refList = [[] for i in range(chrNum)]
for i in range(chrNum):
    reference = "chr21.fa"######################
    refList = read_ref_file(reference, refList, i)
    #print('chrNum',chrNum)
chrLen = np.full(chrNum, 0)
modeList = np.full(chrNum, 0.0)

for i in range(chrNum):
    chrLen[i] = len(refList[i])
    print(chrLen[i])

print("Read bam file:", bam)

ReadCount = np.full((chrNum, np.max(chrLen)), 0)
ReadCount = get_RC(bam, chrList, ReadCount)
sum_k = 0
for i in range(chrNum):
    binNum = int(chrLen[i]/binSize)+1
    col = round(chrLen[i] / 500000)	#一共有48129895个碱基比对上，除以50w等于96。25979
    sum_k += round(col / 10)

    pos, RD = ReadDepth(ReadCount[0], binNum, refList[i])
    for m in range(len(RD)):
        if np.isnan(RD[m]).any():
            RD[m] = (RD[m-1] + RD[m+1]) / 2

    numbin = len(RD)
    modeList[i] = modeRD(RD)

    scalRD = scaling_RD(RD, modeList[i])
    print("segment count...")
    v = robjects.FloatVector(scalRD)
    m = robjects.r['matrix'](v, ncol=col)
    robjects.r.source("CBS_data.R")
    robjects.r.CBS_data(m, segpath)

    num_col = int(numbin / col) + 1
    seg_start, seg_end, seg_count, seg_len = Read_seg_file(num_col, numbin)
    seg_count = np.array(seg_count)
    seg_count, seg_start, seg_end = seg_RD(RD, pos, seg_start, seg_end, seg_count)

    all_RD.extend(seg_count)
    all_start.extend(seg_start)
    all_end.extend(seg_end)
    all_chr.extend(chrList[i] for j in range(len(seg_count)))
    for m in range(len(all_RD)):
        if np.isnan(all_RD[m]).any():
            all_RD[m] = all_RD[m-1]

all_chr = np.array(all_chr)
all_start = np.array(all_start)
all_end = np.array(all_end)
all_RD = np.array(all_RD)


# lof
k = int(sum_k)
print("calculating scores...")
dis, newpos = dis_matrix(all_RD)
dis, min_matrix = k_matrix(dis, k)
min_matrix = min_matrix.astype(np.int)
density = reach_density(dis, min_matrix, k)
density = np.array(density)
scores = get_scores(density, min_matrix, all_RD, k)
print('allRD',len(all_RD))

mode = np.mean(modeList)
#############################
#Write_data_file(all_chr, all_start, all_end, all_RD, scores)
upper = boxplot(scores)
CNV_chr, CNVstart, CNVend, CNVRD, CNVtype = combiningCNV(all_chr, all_start, all_end, all_RD, scores, upper, mode)
CN = calculating_CN(mode, CNVRD, CNVtype)
Write_CNV_File(CNV_chr, CNVstart, CNVend, CNVtype, CN, outfile)
print('upper',upper)
#plot(polar_rd[:,0],polar_rd[:,1])
endtime = datetime.datetime.now()
print("running time: " + str((endtime - starttime).seconds) + " seconds")
print('-------------------------------------------------------------')
