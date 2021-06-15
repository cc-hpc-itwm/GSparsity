import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filepath = './experiments_sparsity_graph.csv'

def plot_graph(input_csv, output_filename):
    df = pd.read_csv(input_csv, sep = ';')
    avg_by_threshold = df.groupby('Pruned_threshold')['Retrain Accuracy','weight_percentage_pruned'].mean()
    print(avg_by_threshold)
    xs = avg_by_threshold['weight_percentage_pruned'] #(100 - avg_by_threshold['weight_percentage_pruned'])/100
    ys = avg_by_threshold['Retrain Accuracy']

    f = plt.figure()
    plt.plot(xs, ys, 'o-')

    for x,y in zip(xs,ys):
        label = "{:.2f}".format(y)
        plt.annotate(label, (x,y))

    #plt.gca().invert_xaxis()
    plt.xticks(np.arange(0, 100, 20))
    plt.grid(True)
    plt.xlabel('percentage of parameters pruned (in %)')
    plt.ylabel('Accuracy (after retraining')
    plt.savefig(output_filename)
    plt.close(f)

if __name__ == '__main__':
    plot_graph(filepath, output_filename='sparsity_and_accuracy_graph.png')