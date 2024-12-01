import matplotlib.pyplot as plt
import numpy as np


def plt_cdf(x, xname, unitname, num_bins=40):
    # use the histogram function to bin the data
    counts, bin_edges = np.histogram(x, bins=num_bins)
    total_nums = len(x)
    # now find the cdf
    cdf = np.cumsum(counts) / total_nums
    # and finally plot the cdf
    plt.figure()
    plt.plot(bin_edges[1:], cdf)
    plt.xlabel(xname + '' + unitname)
    plt.ylabel('Count')
    plt.title(xname + ' CDF')
    plt.savefig('figs/' + xname + ' CDF.png')
    plt.show()


def plt_pdf(x, xname, unitname, num_bins=40):
    # plt.figure()
    plt.hist(x, bins=num_bins, range=(np.min(x), np.max(x)), label=xname)
    plt.xlabel(xname + '' + unitname)
    plt.ylabel('Count')
    plt.title(xname + ' PDF')
    plt.legend()
    plt.savefig('figs/' + xname + ' PDF.png')
    # plt.show()

if __name__ == '__main__':


    pass