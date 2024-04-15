class Counter(object):
    "Model a counter."

    #class variable 
    instances = 0

    #Constructor 
    def __init__(self):
        "Sets up the counter."
        Counter.instances += 1
        self.reset()
    
    #Mutator methods
    def reset(self):
        "Sets the counter to 0."
        self.value = 0
    
    def increment(self, amount = 1):
        "Adds amount to the counter."
        self.value += amount
    
    def decrement(self, amount = 1):
        "Subtracts amount from the counter."
        self.value -= amount
    
    #Accessor methods
    def getValue(self):
        "Returns the counter's value."
        return self.value
    
    def __str__(self):
        "Returns the string representation of the counter."
        return str(self.value)
    
    def __eq__(self, other):
        "Returns True if self equals other or False otherwise."
        if self is other: return True
        if type(self) != type(other): return False
        return self.value == other.value

