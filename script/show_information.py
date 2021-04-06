import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import csv
import pandas as pd

csfont = { 'fontname': 'Times New Roman', 'fontsize': 30 }

tocsv = True

file = "./results/info_stat.pkl"

info = pickle.load( open( file, 'rb' ) )

n_kern = len( info ) + 1
info_mean = np.zeros( n_kern )
info_min = np.zeros( n_kern )
info_max = np.zeros( n_kern )

if tocsv:
    csv_file = "./results/info_stat.csv"
    csv_fields = [ 'kernel', 'batch', 'info' ]
    with open( csv_file, 'w' ) as csvfile:
        writer = csv.DictWriter( csvfile, fieldnames=csv_fields )
        writer.writeheader()
        for kern in info:
            i = 0
            for val in info[ kern ]:
                dict_data = { 'kernel': kern, 'batch': i, 'info': val }
                writer.writerow( dict_data )

                i += 1


i = 1
for kern in info:
    info_mean[ i ] = np.mean( info[ kern ] )
    info_min[ i ] = np.min( info[ kern ] )
    info_max[i ] = np.max( info[ kern ] )

    i += 1

sz_kern = [1, 3, 5, 7, 9, 11]

info_stat = pd.read_csv( "./results/info_stat.csv" )
# info_counts = info_stat.groupby( ['kernel', 'info'] ).size().reset_index( name='counts' )

fig, ax = plt.subplots( figsize=(16,10), dpi=100 )
sns.violinplot( x="kernel", y="info", data=info_stat, ax=ax )
hold = True
plt.plot( np.arange( len(info_mean) )-1, info_mean, 'k-*' )
# sns.boxplot( x="kernel", y="info", data=info_stat )
# sns.stripplot( info_counts.kernel, info_counts.info, size=info_counts.counts*2, ax=ax )

plt.xlabel( 'Kernel Size', **csfont )
plt.xticks( fontsize=20 )
plt.ylabel( 'Information', **csfont )
plt.yticks( fontsize=20 )
plt.grid( axis='y' )

plt.savefig( './results/info_stat.png' )