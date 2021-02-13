from numpy import double
import numpy as np
from sklearn.model_selection import train_test_split

from Point import Point
from rule import rule

epsilon = 0.00001
point_list = []
rules_list = []
best_rules = []
all_success = []

p_train8 = []
p_train7 = []
p_train6 = []
p_train5 = []
p_train4 = []
p_train3 = []
p_train2 = []
p_train1 = []

p8 = []
p7 = []
p6 = []
p5 = []
p4 = []
p3 = []
p2 = []
p1 = []

def slope(x1, y1, x2, y2):
    if (x2 - x1) == 0:
        m = 0
    else:
        m = (y2 - y1) / (x2 - x1)
    return m

def findMidpoint(x1, y1, x2, y2):
    y = (y2 + y1) / 2
    x = (x2 + x1) / 2
    p = Point(x, y, 0)
    return p

# Normalize the points to reach the total weight of 1
def NormalizationWeight(weight, array_weight):
    for i_weight in range(len(array_weight)):
        array_weight[i_weight] = array_weight[i_weight] / weight

def print_rules():
    print("The average of 1 rule: Train: ", np.sum(p_train1)/10, "Test: ", np.sum(p1)/10)
    print("The average of 2 rule: Train: ", np.sum(p_train2)/10, "Test: ", np.sum(p2)/10)
    print("The average of 3 rule: Train: ", np.sum(p_train3)/10, "Test: ", np.sum(p3)/10)

def read_file(path):
    # read the data
    file = open(path, "r")
    # array of lines from data
    fileLines = file.readlines()
    # create points from data
    if str(path).startswith('HC'):
        for line in fileLines:
            line_split = line.split()
            point_list.append(Point(double(line_split[0]), double(line_split[2]), int(line_split[1])))
           # change the label to (-1, 1)
        for point_train in point_list:
            if point_train.label == 2:
                point_train.label = -1
    else:
         for line in fileLines:
            line_split = line.split(",")
            if line_split[4].startswith("Iris-ve"):
                point_list.append(Point(double(line_split[1]), double(line_split[2]), 1))
            elif line_split[4].startswith("Iris-vi"):
                point_list.append(Point(double(line_split[1]), double(line_split[2]), -1))


def adaboost(path, times, best_k):
    for t in range(times):
        print("Iteration: ", t)
        # read the data and create points in point_list
        read_file(path)
        # split the data to train and test
        train, test = train_test_split(point_list, test_size=0.5)
        # total of points
        n_samples = len(train)

        # init the weight to be 1/n
        array_weight = np.full(n_samples, (1 / n_samples))

        # prediction array
        predictions = np.zeros(n_samples)

        true_error = 0
        alpha = 0
        index = 0
        k = 0
        for i in train:
            for j in train:
                true_error = 0
                k = k + 1
                alpha = 0
                if i != j:
                    # slope of equation
                    slope_m = slope(i.x, i.y, j.x, j.y)
                    # finding the midpoint
                    midpoint = findMidpoint(i.x, i.y, j.x, j.y)
                    if slope_m == 0:
                        reverse_slope = 0
                    else:
                        reverse_slope = (-1 / slope_m)
                    # b of equation
                    b_reverse = ((-1 * reverse_slope) * midpoint.x) + midpoint.y
                    # --------------- FIND THE ERROR FOR THIS RULE ---------------
                    for p in train:
                        # check if the point is under the equation
                        y = reverse_slope * p.x + b_reverse
                        # check whether the point is above the line or below
                        # if y > point.y -> the point is below the line
                        if y > p.y:
                            predictions[index] = 1
                        else:
                            # the point is above the line
                            predictions[index] = -1

                        true_error = true_error + array_weight[index] * int(predictions[index] != p.label)
                    if true_error < 0.5:
                        alpha = 0.5 * np.log((1 - true_error + epsilon) / (true_error + epsilon))
                        rules_list.append(rule(reverse_slope, b_reverse, alpha, -1, true_error))
                    for w in range(len(array_weight)):
                        array_weight[w] = array_weight[w] * np.exp((-1* alpha) * (predictions[w] * train[w].label))
                    sum_weight = np.sum(array_weight)
                    NormalizationWeight(sum_weight, array_weight)
        rules_sort = sorted(rules_list, key=lambda rule: rule.alpha, reverse=True)
        for idx in range(best_k):
        # print("idx: ", idx)
        # print("k " , k)
            best_rules.append(rules_sort[idx])
        check_success(1, train, "train")
        check_success(1, test, "test")
        check_success(2, train, "train")
        check_success(2, test, "test")
        check_success(3, train, "train")
        check_success(3, test, "test")

    print_rules()

# -------------- CHECK ON THE TEST --------------
def check_success(k, type, st):
    success = 0
    for i in range(len(type)):
        fx = 0
        for j in range(k):
            sign = 1
            y = best_rules[j].m * type[i].x + best_rules[j].b
            if type[i].y > y:
                sign = -1
            fx = fx + (best_rules[j].alpha * sign)
        if fx > 0:
            guess = 1
        else:
            guess = -1
        if guess == type[i].label:
                success = success + 1
    if k == 1:
        if st == "train":
            p_train1.append(success / len(type))
        else:
            p1.append(success / len(type))
    if k == 2:
        if st == "train":
            p_train2.append(success / len(type))
        else:
            p2.append(success / len(type))
    if k == 3:
        if st == "train":
            p_train3.append(success / len(type))
        else:
            p3.append(success / len(type))


def main():

    # path = "HC_Body_Temperature.txt"
    path = "iris.data"
    # read_file(path, "HC")
    adaboost(path, 10, 8)

    # path = "iris.data"
    # read_file(path, "iris")




if __name__ == '__main__':
    main()

