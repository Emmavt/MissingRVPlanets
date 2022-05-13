"""
Script to take in RV data and standardize the format for use in other modules. To be run from the command line.
Inputs: file name to check, and (optional) name of telescope/instrument
Outputs: csv file with standardized data format to input else where
"""
import pandas as pd
import sys
from astropy.time import Time

# Save input from command line argument
file = sys.argv[0]
# Check if file is a txt or csv file
if file.endswith('.txt') or file.endswith('.csv'):
    data = pd.read_csv(file, sep=',', skiprows=14)
else:
    print('Please input .txt or .csv file!')

# Check if there is a second input, and if so, set it as the telescope parameter
if len(sys.argv) > 0:
    facility = sys.argv[1]
    data['tel'] = facility

# Check we have the correct number of columns + data
assert data.shape[1] == 4, "Incorrect number of columns - 4 required (time, mnvel, errvel, tel)"
# Standardize column labels
data.columns = ['time', 'mnvel', 'errvel', 'tel']

# Check time formatting and standardize + convert to Julian Dates
if type(data['time'][0]) == str:
    data['time'] = pd.DatetimeIndex(pd.to_datetime(data['time'], utc=True)).to_julian_date()
else:
    # This is a method used in Ivshina & Winn 2022 to search for time series data on the arXiv, and seemed appropriate to add here.
    assert data['time'][0]-2E6 >= 0, "Check time formatting manually, may not be in JD"

# Save correctly formatted file with _st added to name
data.to_csv(file.split('.')[0]+'_st.'+file.split('.')[1], index=False)
