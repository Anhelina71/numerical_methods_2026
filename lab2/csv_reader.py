import csv

def read_data(filename):
    x = []
    y = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    return x, y
