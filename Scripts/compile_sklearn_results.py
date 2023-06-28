import numpy as np
import pandas as pd
from redshift_utils import plot_hex, outlier_rate, norm_residual, norm_std_dev, norm_mad

seed_list = [
        28038,243145,205639,137877,136035,269736,27000,219084,63293,29586,165398,
        306500,231143,239118,225401,246949,161234,40248,263814,141492,157190,116489,
        57349,291151,245536,202276,126422,258477,171351,139302,141515,71389,28945,
        174227,278938,20048,269639,260007,86946,198443,51908,238160,220075,111377,
        21337,304953,140016,280582,212974,244536,238729,61147,114324,146624,156385,
        13761,171709,48471,233538,214585,289820,233973,115184,303951,129072,102360,
        284482,116383,23983,147515,249970,59524,145436,40816,215663,149446,103734,
        71285,177342,210428,295405,137335,50481,261593,197846,219994,30544,98132,
        241225,261461,136727,252823,264425,121729,282142,90580,75259,214412,200044,43904
        ]

out_rate = []
sigma = []
nmad = []

for i in range(100):
    file_name = "seed_" + str(i) + "/predictions.csv"
    try:
        predictions = np.loadtxt(file_name, delimiter=',', skiprows=1)  
        out_rate.append(outlier_rate(norm_residual(np.array(predictions[:,0]), np.array(predictions[:,1]))))
        sigma.append(norm_std_dev(norm_residual(np.array(predictions[:,0]), np.array(predictions[:,1]))))
        nmad.append(norm_mad((np.array(predictions[:,0]) - np.array(predictions[:,1]))))
    except:
        print("Seed: " + str(seed_list[i]) + " (" + str(i) + ") missing predictions.")
print("Outlier    Std Err    NumComplete")
print(np.mean(out_rate), (np.std(out_rate) / np.sqrt(len(out_rate))), len(out_rate))

print("NormSTD    Std Error")
print(np.mean(sigma), (np.std(sigma) / np.sqrt(len(sigma))))

print("NMAD    Std Error")
print(np.mean(nmad), (np.std(nmad) / np.sqrt(len(sigma))))
