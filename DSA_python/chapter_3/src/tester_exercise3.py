import timeit

def test_algorithm():
    print("Hello World")

if __name__ == '__main__':
    print(timeit.timeit("test_algorithm()", setup="from __main__ import test_algorithm"))