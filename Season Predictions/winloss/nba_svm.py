'''
Support vector regression for NBA team win/loss percentage
Data taken from basketballreference
use sys.argv[1] = teamstat csv file

Author: Aaron Hunt
'''

import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import sys


dates = []
spreads = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile  )
		next(csvFileReader) # First row is a header
		for row in csvFileReader:
			if row[1] != '':
				dates.append(int(row[1].split('-')[0]))
				spreads.append(float(row[7]))
	return

def predict_spread(dates, spreads):
	dates = np.reshape(dates, (len(dates), 1)) # reshape to nx1 np array

	svr = SVR(C=1) 
	svr_linear = SVR(kernel='linear')

	svr_linear.fit(dates, spreads)
	svr.fit(dates, spreads)
	p = svr.predict(dates)
	p_linear = svr_linear.predict(dates)

	prediction = svr.predict(73)
	prediction_linear = svr_linear.predict(73)

	print('2018-2019 team win/loss ratio prediction (RBF):', prediction[0])

	plt.scatter(dates, spreads, color='black', label='Data')

	plt.plot(dates, p, color='red', label='RBF model')
	plt.plot(dates, p_linear, color='green', label='linear model')
	plt.title('Support Vector Regression: Bulls')
	plt.xlabel('Year')
	plt.ylabel('Win/Loss Ratio')
	plt.legend()
	plt.show()
	plt.savefig('bulls_winloss')

	return prediction[0], prediction_linear[0]


if __name__ == '__main__':
	np.random.seed(10)
	get_data(sys.argv[1])
	predict_spread(dates, spreads)

	print('Complete!')

