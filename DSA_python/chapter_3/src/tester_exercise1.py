from counter import Counter

def algorithm(problemSize, counter):
    while problemSize > 0:
        counter.increment()
        problemSize = problemSize //2
    return counter
   
    
if __name__ == "__main__":
    problemSize = 100
    counter = Counter()
    algorithm(problemSize,counter)
    print("%12d%15s" % (problemSize, counter))

