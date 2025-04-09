"""
Description     : Simple Python implementation of the Apriori Algorithm
Modified from:  https://github.com/asaini/Apriori
Usage:
    $python apriori.py -f DATASET.csv -s minSupport
    $python apriori.py -f DATASET.csv -s 0.15
"""

import sys
import os
import time
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser

##--------------------------------------------------------------------------------------------------
### Some funtion usage ###

def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)
    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1
    for item, count in localSet.items():
        support = float(count) / len(transactionList)
        if support >= minSupport:
            _itemSet.add(item)
    return _itemSet


def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
    return itemSet, transactionList


def getSupport(item, freqSet, transactionList):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

##--------------------------------------------------------------------------------------------------
### Apriori mining process ###
def runApriori(data_iter, minSupport):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
    """
    start_time = time.time()
    itemSet, transactionList = getItemSetTransactionList(data_iter)
    freqSet = defaultdict(int)
    largeSet = dict()  
    num_before_pruning = 0
    num_after_pruning = 0
    iteration_stats = []

    k = 1
    num_before_pruning = len(itemSet)
    oneCSet= returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    num_after_pruning = len(oneCSet)
    currentLSet = oneCSet
    iteration_stats.append((k, num_before_pruning, num_after_pruning))

    k = 2
    while currentLSet != set([]): 
        print(f"-------------------- Counting k = {k} --------------------")   
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        num_before_pruning = len(currentLSet)
        currentCSet= returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet
        )
        num_after_pruning = len(currentCSet)
        iteration_stats.append((k, num_before_pruning, num_after_pruning))
        currentLSet = currentCSet
        k = k + 1
    
    end_time = time.time()
    exec_time = round(end_time - start_time, 4)
    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item, freqSet, transactionList)) for item in value])
    return toRetItems, iteration_stats, exec_time

### Apriori mining process for closed frequent sets ###
def runApriori_closed(data_iter, minSupport):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
    """
    start_time = time.time()
    itemSet, transactionList = getItemSetTransactionList(data_iter)
    freqSet = defaultdict(int)
    largeSet = dict()  # Global dictionary which stores (key=n-itemSets,value=support) which satisfy minSupport
    closedSet = []

    k = 1
    oneCSet= returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    currentLSet = oneCSet
    
    k = 2
    while currentLSet != set([]): 
        print(f"-------------------- Counting k = {k} --------------------")   
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet= returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet
        )
        ### Find closed frequent sets ###
        num_old = len(closedSet)
        for item in largeSet[k - 1]:
            is_closed = True
            for superSet in currentCSet:
                if item.issubset(superSet) and freqSet[item] == freqSet[superSet]:
                    is_closed = False
                    break
            if is_closed:
                closedSet.append((tuple(item), getSupport(item, freqSet, transactionList)))
        num_new = len(closedSet)
        # print(f"num_old = {num_old}\tnum_new = {num_new}\tadding = {num_new - num_old}")
        currentLSet = currentCSet
        k = k + 1
    
    end_time = time.time()
    exec_time = round(end_time - start_time, 4)
    return closedSet, exec_time

##--------------------------------------------------------------------------------------------------
### Reading and Writing data ###

def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    with open(fname, "r") as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(" ") 
            record = frozenset(line.split(" ")[3:])
            yield record


def printResults(items):
    """prints the generated itemsets sorted by support """
    for item, support in sorted(items, key=lambda x: x[1]):
        print("item: %s , %.3f" % (str(item), support))


def writeResults_task1_file1(items, filename="file1.txt"):
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for item, support in sorted(items, key=lambda x: x[1], reverse=True):
            support_percentage = round(support * 100, 1)
            item_str = "{" + ", ".join(map(str, item)) + "}"
            f.write(f"{support_percentage}\t{item_str}\n")


def writeResults_task1_file2(items, iteration_stats, filename="file2.txt"):
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"{len(items)}\n")
        for item in iteration_stats:
            f.write(f"{item[0]}\t{item[1]}\t{item[2]}\n")


def writeResults_task2_file(items, filename="file.txt"):
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"{len(items)}\n")
        for item, support in sorted(items, key=lambda x: x[1], reverse=True):
            support_percentage = round(support * 100, 1)
            item_str = "{" + ", ".join(map(str, item)) + "}"
            f.write(f"{support_percentage}\t{item_str}\n")


def writeResults_time(exec_time, filename, prefix):
    print(f"Time = {exec_time} sec\n")
    with open(filename, "a") as f:
        f.write(f"{prefix}\t\tTime = {exec_time} sec\n\n")

##--------------------------------------------------------------------------------------------------
### Main function ###

if __name__ == "__main__":

    optparser = OptionParser()
    optparser.add_option(
        "-f", 
        "--inputFile", 
        dest="input", 
        help="filename containing csv", 
        default='A.csv',
    )
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.1,
        type="float",
    )
    optparser.add_option(
        "-t",
        "--taskNum",
        dest="taskNum",
        help="task number in spec",
        default=2,
        type="int",
    )
    (options, args) = optparser.parse_args()

    ### Reading input file ###
    inFile = None
    if options.input is None:
        inFile = sys.stdin
    elif options.input is not None:
        inFile = dataFromFile(options.input)
    else:
        print("No dataset filename specified, system with exit\n")
        sys.exit("System will exit")
    
    minSupport = options.minS
    minSupportName = str(round(minSupport * 100, 1))
    datasetName = str(options.input).split('.')[0]
    taskName = "task" + str(options.taskNum)
    print(f"***** Doing task : {taskName}_{datasetName}_{minSupportName} *****\n")

    ### Doing task 1/2 ###
    if options.taskNum == 1:
        items, iteration_stats, exec_time= runApriori(inFile, minSupport)
        filename1 = "step2_" + taskName + "_" + datasetName + "_" + minSupportName + "_" + "result1.txt"
        filename2 = "step2_" + taskName + "_" + datasetName + "_" + minSupportName + "_" + "result2.txt"
        writeResults_task1_file1(items, filename1)
        writeResults_task1_file2(items, iteration_stats, filename2)
        writeResults_time(exec_time, "step2_time.txt", filename1)

    elif options.taskNum == 2:
        closedSet, exec_time = runApriori_closed(inFile, minSupport)
        filename_closed = "step2_" + taskName + "_" + datasetName + "_" + minSupportName + "_" + "result1.txt"
        writeResults_task2_file(closedSet, filename_closed)
        writeResults_time(exec_time, "step2_time.txt", filename_closed)