from numpy import ma

dataset = ma.array([[1, 2], [3, 4], [5, 6]], mask=[[True, False], [True, True], [False, False]])

for index, row in enumerate(dataset):
    print(index, ": ", row)
    print(type(row))
