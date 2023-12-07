# Question 1 - Midterm 2
# GOPH419
# author: Annika Richardson
import pandas as pd 
import numpy as np
from midterm2.functions import cubic_spline
import matplotlib.pyplot as plt

def main():
    # load data from .xlsx 
    df = pd.read_excel('data/GOPH_419_F2023_Midterm_2_DATA.xlsx', skiprows= 45)
    df = df.drop(['unc'], axis=1)
    # convert to np.array
    df_numpy = np.array(df)
    # slice data to get 
    data = np.array(df_numpy[51:62,:])
    print(f"CO2 data (2010- 2020) =\n {data}\n")
    
    xd = data[:,0]
    yd = data[:,1]

    a, b, c, d = cubic_spline(xd, yd)

    size = len(a)
    s = np.zeros(shape = (size - 1, 1))
    # get spline function for 
    x = 2015.25
    i = 5
    s[i] = a[i] + b[i] * (x - xd[i]) + c[i] * (x - xd[i]) ** 2 + d[i] * (x - xd[i]) ** 3

    print(f"si(2015.25) = {s[i]}")
    plt.plot(xd, yd, 'kd', label = 'mean annual CO2 concentration')
    plt.plot(x, s[i], 'ro', label = f's[5] = {np.round(s[i], 2)}')
    plt.xlabel('year')
    plt.ylabel('CO2 Concentration')
    plt.title('Atmospheric CO2 Concentration at Mauna Loa')
    plt.legend()
    #plt.show()

    plt.savefig('figures/check_interpolation_march2015')

if __name__ == "__main__":
    main()
