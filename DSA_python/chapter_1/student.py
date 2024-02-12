from typing import Any


class Student(object):
    #class variable 
    instances = 0

    #Constructor 
    def __init__(self):
        self.name = None
        self.scores = []
    
    def set_name(self, name):
        self.name = name

    def add_score(self, score):
        self.scores.append(score)
    
    def get_score(self, score_index):
        if score_index < 0 or score_index >= len(self.scores):
            return "Invalid index. Please enter a number between 0 and the number of scores - 1."
        return self.scores[score_index]
    
    def get_name(self):
        return self.name

    def get_number_of_scores(self):
        return len(self.scores)
    
    def update_score(self, index, new_score):
        if index < 0 or index >= len(self.scores):
            return "Invalid index. Please enter a number between 0 and the number of scores - 1."
        self.scores[index] = new_score
    
    def get_average(self):
        return sum(self.scores) / len(self.scores)
    
    def summary(self):
        return f'Name: {self.name}\nScores: {self.scores}\nAverage: {self.get_average()}'

    def __str__(self):
        return self.summary()

Nikhil = Student()
Nikhil.set_name("Nikhil")
Nikhil.add_score(100)
Nikhil.add_score(90)
Nikhil.add_score(80)
Nikhil.update_score(1, 95)
print(f'Name: {Nikhil.get_name()}')
for i in range(Nikhil.get_number_of_scores()):
    print(f'Score {i+1} : {Nikhil.get_score(i)}')