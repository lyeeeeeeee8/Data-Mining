import sys
import os
import time
from optparse import OptionParser
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    transaction_list = []
    with open(fname, "r") as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(" ")  # strip the line by " "
            record = line.split(" ")[3:]     # take the data from 3rd column
            transaction_list.append(record)
    return transaction_list


def writeResults(df_res, filename="file1.txt"):
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for index, row in df_res.iterrows():
            row_support = str(round(row['support'] * 100, 1))
            row_itemsets = "{" + ", ".join(map(str, row['itemsets'])) + "}"
            f.write(f"{row_support}\t{row_itemsets}\n")


def writeResults_time(exec_time, filename, prefix):
    print(f"Time = {exec_time} sec\n")
    # with open(filename, "a") as f:
    #     f.write(f"{prefix}\tTime = {exec_time} sec\n\n")
    
##--------------------------------------------------------------------------------------------------

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
    filename = "step3_task1_" + datasetName + "_" + minSupportName + "_" + "result1.txt"
    print(f"***** Doing task : {datasetName}_{minSupportName} *****")


    ### Data Mining ###
    start_time = time.time()
    te = TransactionEncoder()
    te_ary = te.fit(inFile).transform(inFile)                  # encode the data
    df = pd.DataFrame(te_ary, columns=te.columns_)             # transform to dataframe
    df_res = fpgrowth(df, minSupport, use_colnames=True)       # use fpgrowth to mine
    df_res = df_res.sort_values(by='support', ascending=False) # sort by descending order
    end_time = time.time()
    exec_time = round(end_time - start_time, 4)


    ### Writing result ###
    writeResults(df_res, filename)
    writeResults_time(exec_time, "step3_time.txt", filename)
    

