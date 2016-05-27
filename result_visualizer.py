from matplotlib import pyplot as plt
import pandas as pd

def get_best_results(file_name):
    with open(file_name) as data_file:
        data = json.load(data_file)
    print data

def visualize_results(file_name):
    get_best_results(file_name)
