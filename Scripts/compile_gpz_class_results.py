import numpy as np
import pandas as pd
from redshift_utils import plot_hex, outlier_rate, norm_residual, norm_std_dev, norm_mad
from sklearn.metrics import accuracy_score

def rebin_data(spec_z, pred_z, bin_edges, bin_medians):
    """Function to bin the data
    Args:
        spec_z (np.array): 1-d numpy array holding the measured redshifts to be binned
        pred_z (np.array): 1-d numpy array holding the estimated redshifts to be binned
        bin_edges (np.array): 1-d numpy array holding the edges of the bins
        spec_z (np.array): 1-d numpy array holding the medians of the bins
    Returns:
        np.array: List containing the binned measured redshifts
        np.array: List containing the binned estimated redshifts
    """
    for i in range(1, len(bin_medians) + 1):
        if i < len(bin_medians):
            if i == 1:
                spec_z[np.where((spec_z < bin_edges[i]))[0]] = bin_medians[i-1]
                pred_z[np.where((pred_z < bin_edges[i]))[0]] = bin_medians[i-1]
            else:
                spec_z[np.where((spec_z >= bin_edges[i-1]) &\
                     (spec_z < bin_edges[i]))[0]] = bin_medians[i-1]
                pred_z[np.where((pred_z >= bin_edges[i-1]) &\
                     (pred_z < bin_edges[i]))[0]] = bin_medians[i-1]
        else: 
            spec_z[np.where((spec_z >= bin_edges[i-1]))[0]] = bin_medians[i-1]
            pred_z[np.where((pred_z >= bin_edges[i-1]))[0]] = bin_medians[i-1]
    return spec_z, pred_z

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
out_sigma = []
sigma = []
nmad = []
acc = []

out_rate_mod = []
sigma_mod = []
nmad_mod = []
out_sigma_mod = []
acc_mod = []

for i in range(100):
    bin_edges_name = f"seed_{seed_list[i]}/bin_edges.csv"
    bin_median_name = f"seed_{seed_list[i]}/bin_median.csv"
    bin_edges = pd.read_csv(bin_edges_name)
    bin_edges = np.array(bin_edges.loc[:,"Bin_Edges"])
    bin_median = pd.read_csv(bin_median_name)
    bin_median = np.array(bin_median.loc[:,"Bin_Median"])
    file_name = f"seed_{seed_list[i]}/predictions.csv"
    predictions = pd.read_csv(file_name) 

    z_spec = np.array(predictions["True_z"])
    z_photo = np.array(predictions["Pred_z"])

    z_spec, z_photo = rebin_data(z_spec, z_photo, bin_edges, bin_median)

    out_rate.append(outlier_rate(norm_residual(z_spec, z_photo)))
    sigma.append(norm_std_dev(norm_residual(z_spec, z_photo)))
    nmad.append(norm_mad((z_spec - z_photo)))
    out_sigma.append(outlier_rate(norm_residual(z_spec, z_photo), (2*sigma[i])))
    acc.append(accuracy_score(z_spec.astype(str), z_photo.astype(str), normalize=True))


    # Modified
    z_spec_mod = np.copy(z_spec)
    z_photo_mod = np.copy(z_photo)

    # z_spec_mod[np.where(z_spec_mod == bin_median[-1])] = bin_median[-2] + ((0.149*bin_median[-2])/(1+bin_median[-1]))
    # z_photo_mod[np.where(z_photo_mod == bin_median[-1])] = bin_median[-2] + ((0.149*bin_median[-2])/(1+bin_median[-1]))
    z_spec_mod[np.where(z_spec_mod == bin_median[-1])] = bin_median[-2] 
    z_photo_mod[np.where(z_photo_mod == bin_median[-1])] = bin_median[-2] 

    out_rate_mod.append(outlier_rate(norm_residual(z_spec_mod, z_photo_mod)))
    sigma_mod.append(norm_std_dev(norm_residual(z_spec_mod, z_photo_mod)))
    nmad_mod.append(norm_mad((z_spec_mod - z_photo_mod)))
    out_sigma_mod.append(outlier_rate(norm_residual(z_spec_mod, z_photo_mod), (2*sigma[i])))
    acc_mod.append(accuracy_score(z_spec_mod.astype(str), z_photo_mod.astype(str), normalize=True))


print("Outlier    Std Err    NumComplete")
print(np.mean(out_rate), (np.std(out_rate) / np.sqrt(len(out_rate))), len(out_rate))

print("NormSTD    Std Error")
print(np.mean(sigma), (np.std(sigma) / np.sqrt(len(sigma))))

print("NMAD    Std Error")
print(np.mean(nmad), (np.std(nmad) / np.sqrt(len(nmad))))


print("Out2Sig    Std Error")
print(np.mean(out_sigma), (np.std(out_sigma) / np.sqrt(len(out_sigma))))


print("Accuracy    Std Error")
print(np.mean(acc), (np.std(acc) / np.sqrt(len(acc))))

print("############################\nModified\n#############################")

print("Outlier    Std Err    NumComplete")
print(np.mean(out_rate_mod), (np.std(out_rate_mod) / np.sqrt(len(out_rate_mod))), len(out_rate_mod))

print("NormSTD    Std Error")
print(np.mean(sigma_mod), (np.std(sigma_mod) / np.sqrt(len(sigma_mod))))

print("NMAD    Std Error")
print(np.mean(nmad_mod), (np.std(nmad_mod) / np.sqrt(len(nmad_mod))))


print("Out2Sig    Std Error")
print(np.mean(out_sigma_mod), (np.std(out_sigma_mod) / np.sqrt(len(out_sigma_mod))))


print("Accuracy    Std Error")
print(np.mean(acc_mod), (np.std(acc_mod) / np.sqrt(len(acc_mod))))