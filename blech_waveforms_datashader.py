# Import stuff
import datashader as ds
import datashader.transfer_functions as tf
from functools import partial
from datashader.utils import export_image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import imread

# A function that accepts a numpy array of waveforms and creates a datashader image from them
def waveforms_datashader(waveforms, x_values):

	# Make a pandas dataframe with two columns, x and y, holding all the data. The individual waveforms are separated by a row of NaNs
	dfs = []
	split = pd.DataFrame({'x': [np.nan]})
	for i in range(waveforms.shape[0]):
    		x = x_values
    		y = waveforms[i, ::10] # Downsample the waveforms 10 times (to remove the effects of 10 times upsampling during de-jittering)
    		df = pd.DataFrame({'x': x, 'y': y})
    		dfs.append(df)
    		dfs.append(split)

	df = pd.concat(dfs, ignore_index=True)

	# Datashader function for exporting the temporary image with the waveforms
	export = partial(export_image, background = "white", export_path="datashader_temp")

	# Produce a datashader canvas
	canvas = ds.Canvas(x_range = (np.min(x_values), np.max(x_values)), 
			   y_range = (df['y'].min() - 10, df['y'].max() + 10),
			   plot_height=1200, plot_width=1600)
	# Aggregate the data
	agg = canvas.line(df, 'x', 'y', ds.count())   
	# Transfer the aggregated data to image using log transform and export the temporary image file
	export(tf.shade(agg, how='eq_hist'),'tempfile')

	# Read in the temporary image file
	img = imread('datashader_temp/tempfile.png')
	
	# Figure sizes chosen so that the resolution is 100 dpi
	fig,ax = plt.subplots(1, 1, figsize = (8,6), dpi = 200)
	# Start plotting
	ax.imshow(img)
	# Set ticks/labels - 10 on each axis
	ax.set_xticks(np.linspace(0, 1600, 10))
	ax.set_xticklabels(np.floor(np.linspace(np.min(x_values), np.max(x_values), 10)))
	ax.set_yticks(np.linspace(0, 1200, 10))
	ax.set_yticklabels(np.floor(np.linspace(df['y'].max() + 10, df['y'].min() - 10, 10)))

	# Delete the dataframe
	del df

	# Return and figure and axis for adding axis labels, title and saving the file
	return fig, ax


		 

	
