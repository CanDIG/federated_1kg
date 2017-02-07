#!/usr/bin/env python
import csv
import matplotlib.pylab as plt
import matplotlib.colors as color

def population_to_colors(populations):
    pop_to_rgb = { 'ACB': (0.84, 0.52, 0.13, 1.0), 'GWD': (0.96, 0.92, 0.18, 1.0),
                   'BEB': (0.37, 0.07, 0.43, 1.0), 'PEL': (0.71, 0.02, 0.1, 1.0),
                   'LWK': (0.72, 0.6, 0.3, 1.0), 'MSL': (0.8, 0.67, 0.15, 1.0),
                   'GBR': (0.48, 0.72, 0.79, 1.0), 'IBS': (0.35, 0.43, 0.66, 1.0),
                   'ASW': (0.77, 0.32, 0.11, 1.0), 'TSI': (0.12, 0.13, 0.32, 1.0),
                   'KHV': (0.39, 0.64, 0.22, 1.0), 'CEU': (0.17, 0.23, 0.53, 1.0),
                   'SAS': (0.52, 0.27, 0.54, 1.0), 'EAS': (0.67, 0.76, 0.15, 1.0),
                   'AMR': (0.45, 0.13, 0.11, 1.0), 'YRI': (0.92, 0.75, 0.36, 1.0),
                   'CHB': (0.67, 0.77, 0.16, 1.0), 'CLM': (0.62, 0.14, 0.16, 1.0),
                   'CHS': (0.45, 0.67, 0.19, 1.0), 'ESN': (0.94, 0.77, 0.14, 1.0),
                   'FIN': (0.39, 0.68, 0.74, 1.0), 'AFR': (0.97, 0.92, 0.24, 1.0),
                   'GIH': (0.32, 0.19, 0.5, 1.0), 'PJL': (0.69, 0.0, 0.45, 1.0),
                   'EUR': (0.53, 0.73, 0.84, 1.0), 'STU': (0.5, 0.25, 0.54, 1.0),
                   'MXL': (0.69, 0.0, 0.16, 1.0), 'ITU': (0.53, 0.13, 0.31, 1.0),
                   'CDX': (0.32, 0.56, 0.2, 1.0), 'JPT': (0.25, 0.49, 0.2, 1.0),
                   'PUR': (0.62, 0.14, 0.16, 1.0)}

    if type(populations) is list:
        colors = [ pop_to_rgb[pop] for pop in populations ] 
    else:
        colors = pop_to_rgb[populations]

    return colors

def makeplot():
    # popdiffs data, taken from fig 2a
    pops = ["BEB", "PJL", "STU", "ITU", "GIH", "TSI", "IBS", "GBR", "CEU",
            "FIN", "CHS", "CHB", "KHV", "CDX", "JPT", "PUR", "CLM", "MXL",
            "PEL", "ASW", "ACB", "YRI", "ESN", "GWD", "MSL", "LWK"]
    values = [6.40,  9.35, 11.80, 12.77, 25.09,  0.88,  0.86,  1.34,  2.30,
              35.37,  7.70, 13.11, 26.43, 30.86, 37.75,  1.68,  5.12, 20.41,
              67.80,  6.55, 10.48, 27.75, 61.80, 74.13, 96.83, 177.80]

    colors = population_to_colors(pops)

    plt.barh(range(len(values), 0, -1), values, color=colors, tick_label=pops, edgecolor='k')
    plt.show()

if __name__ == "__main__":
    makeplot()
