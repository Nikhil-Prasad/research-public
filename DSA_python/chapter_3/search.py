def indexOfMin(lyst):
    "Returns the index of the minimum item"
    minIndex = 0 
    currentIndex = 1 
    while currentIndex < len(lyst):
        if lyst[currentIndex] < lyst[minIndex]:
            minIndex = currentIndex
        currentIndex += 1 
    return minIndex

def sequentialSearch(target,lyst):
    """
    Returns the position of the target item if found or -1 otherwise
    """
    position = 0
    while position < len(lyst):
        if target == lyst[position]:
            return position 
        position += 1 
    return -1 

def binarySearch(target,sortedLyst):
    left = 0 
    print("left:", left)
    right = len(sortedLyst) - 1
    print("right:", right) 
    while left <= right:
        midpoint = (left + right)//2
        print("midpoint:", midpoint)
        if target == sortedLyst[midpoint]: 
            return midpoint
        elif target < sortedLyst[midpoint]:
            right = midpoint -1 
            print("right in loop:", right)
        else: 
            left = midpoint + 1 
            print('left in loop:', left)
    print("value not found")
    return -1 

test = [20,44,48,55,62,66,74,88,93,99]
target = 90

binarySearch(target,test)