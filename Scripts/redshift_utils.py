# from astropy.table import Table
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.colors import LogNorm
from typing import Tuple
try:
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from tqdm import tqdm
except:
    pass

def read_fits(filename, colnames):
    """Read an astronomical table, outputting specific columns. 

    Args:
        filename (String): Name of file to be read
        colnames ([String]): Array of columns to be read. 

    Returns:
        table_data (Astropy Table): Astropy table of data
    """
    # table_data = Table.read(filename)

    hdul = fits.open(filename)
    hdulData = hdul[1].data
    #Create catalogueData array from the redshift column
    table_data = np.reshape(np.array(hdulData.field(colnames[0]), dtype=np.float32), [len(hdulData.field(colnames[0])),1])
    #Add the columns required for the test
    for i in range(1, len(colnames)):
        table_data = np.hstack([table_data,np.reshape(np.array(hdulData.field(colnames[i]), dtype=np.float32), [len(hdulData.field(colnames[i])),1])])
    return table_data


def split_data(x_vals, y_vals, rand_generator=None, split_rate=0.3, field_one=None, field_two=None, field_list=None):
    """Splits large data set into training and test sets.

    Args:
        x_vals (Numpy Array (2-D)): 2-D Array holding the x-vals
        y_vals (Numpy Array (1-D)): 1-D Array holding the y-vals
        rand_generator (Numpy Random Generator, optional): Random Generator to use to generate 70-30 Training-Testing Split. Defaults to None.
        field_one (String, optional): Field to use as the training set. Defaults to None.
        field_two (String, optional): Field to use as the test set. Defaults to None.
        field_list (Numpy Array (1-D), optional): Array holding the field each galaxy belongs in. Defaults to None.

    Returns:
        Numpy Arrays: The split Training and Test sets. 
    """
    if rand_generator is not None:
        try:
            rand_gen = np.random.default_rng(rand_generator)
            test_indices = rand_gen.choice(len(x_vals), round(len(x_vals)*split_rate), replace=False)
        except:
            test_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.3), replace=False)
        train_indices = np.array(list(set(range(len(x_vals))) - set(test_indices)))
        x_vals_train = x_vals[train_indices]
        x_vals_test = x_vals[test_indices]
        y_vals_train = y_vals[train_indices]
        y_vals_test = y_vals[test_indices]
    else:
        x_vals_test = x_vals[np.where(field_list == field_one)[0]]
        y_vals_test = y_vals[np.where(field_list == field_one)[0]]
        x_vals_train = x_vals[np.where(field_list == field_two)[0]]
        y_vals_train = y_vals[np.where(field_list == field_two)[0]]

    return x_vals_train, x_vals_test, y_vals_train, y_vals_test

def split_train_valid_test(x_vals:np.ndarray, y_vals:np.ndarray, seed:int, splits=[0.3,0.2857142857]) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Splits large data set into training and test sets.

    Args:
        x_vals (Numpy Array (2-D)): 2-D Array holding the x-vals
        y_vals (Numpy Array (1-D)): 1-D Array holding the y-vals
        seed (int): Random seed to use in the split.
        splits (list; optional): 2-element list containing the splits to use. First element is test ratio, second element splits validation set. Defaults to 50/20/30 (training/validation/test)
    Returns:
        Numpy Arrays: The split Training and Test sets. 
    """
    # Default ratios: 30% for test set (leaving 70% for training/validation). 70% is then split 
    # into 50/20%. That second calculation is x = ((validation set size)/(remaining size))
    # -> x = 0.2/0.7 -> 0.2857142857

    # Split the test set out
    x_train, x_test, y_train, y_test = train_test_split(\
        x_vals, y_vals, test_size=splits[0], random_state=seed)

    # Now working out the training and validation
    x_train, x_val, y_train, y_val = train_test_split(\
        x_train, y_train, test_size=splits[1], random_state=seed)

    return x_train, x_val, x_test, y_train, y_val, y_test

def norm_mad(data, axis=None):
    """Calculate the Median Absolute Deviation

    Args:
        data ([float]): Array of values to calculate the MAD of. 
        axis (int, optional): Which axis (if using 2+-d Array). Defaults to None.

    Returns:
        Median Absolute Deviation, calculated over the provided axis
    """
    return (1.4826 * np.median(np.absolute(data - np.median(data, axis)), axis))

def outlier_rate(resid, out_frac=0.15):
    """Calculate the outlier rate fraction

    Args:
        resid (array): Array of residuals, normalised by redshift. 
        out_frac (decimal): Value to use as the outlier cutoff. 

    Returns:
        [type]: [description]
    """
    outlier=100*len(resid[np.where(abs(resid)>out_frac)])/len(resid)
    return outlier

def norm_std_dev(resid):
    """Calculate the normalised standard deviation

    Args:
        resid (array): Array of residuals, normalised by redshift. 
        out_frac (decimal): Value to use as the outlier cutoff. 

    Returns:
        [type]: [description]
    """
    sigma=np.std(resid)
    return sigma

def norm_residual(spec_z, pred_z):
    """Calculate the outlier rate fraction

    Args:
        spec_z (array): Measured redshifts
        pred_z (array): Predicted redshifts
        out_frac (decimal): Value to use as the outlier cutoff. 

    Returns:
        [type]: [description]
    """
    residual=(spec_z-pred_z)/(1+spec_z)
    return residual

def plot_hex(spec_z, pred_z, file_name=None, title=None, c_map=None):
    plt.rcParams['font.size'] = 16
    if c_map is None:
        c_map = "hot"
    else:
        c_map = "hot_r"
    
    residual=norm_residual(spec_z,pred_z) 
    out_num = outlier_rate(residual, 0.15)
    sigma=norm_std_dev(residual)
    nmad=norm_mad(residual)
    
    fig, [ax1,ax2] = plt.subplots(2, sharex=True,  gridspec_kw = {'height_ratios':[2, 1]},figsize=(10,15))
    cbar = ax1.hexbin(np.array(spec_z), pred_z, norm=LogNorm(), cmap=c_map,mincnt=1)#, extent=(0,7,0,7), yscale=log_y, xscale=log_x)

    xlab=.2
    ylab=6.7
    step=-.3
    ax1.text(xlab, ylab, r'$N='+str(spec_z.shape[0])+'$')
    ax1.text(xlab, ylab+ step, r'$\sigma='+str(round(sigma, 2))+r'$')
    ax1.text(xlab, ylab+ 2*step,        r'$NMAD='+str(round(nmad, 2))+r'$')
    ax1.text(xlab, ylab+ 3*step,        r'$\eta='+str(round(out_num, 2))+r'\%$')


    
    ax1.plot([0,7],[0,7], 'r--',linewidth=1.5)
    ax1.plot([0,7],[0.15,8.2], 'b--',linewidth=1.5)
    ax1.plot([0,7],[-.15,5.8], 'b--',linewidth=1.5)
    ax1.set_xlim(0,7)
    ax1.set_ylim(0,7)
    ax1.grid()
    fig.colorbar(cbar, ax=ax1)
    ax1.set_ylabel('$z_{photo}$')
    ax1.set_xlabel('$z_{spec}$')
    
    ax2.plot([0,7],[0,0], 'r--',linewidth=1.5)
    ax2.plot([0,7],[0.15,.15], 'b--',linewidth=1.5)
    ax2.plot([0,7],[-.15,-.15], 'b--',linewidth=1.5)
    cbar2 = ax2.hexbin(np.array(spec_z), residual, mincnt=1, cmap=c_map, extent=(0,7,-1,1), norm=LogNorm())
    fig.colorbar(cbar2, ax=ax2)
    ax2.grid()
    ax2.axis([0,7,-.5, .5])
    ax2.set_ylabel(r'$\frac{z_{spec}-z_{photo}}{z_{spec}+1}$')
    ax2.set_xlabel('$z_{spec}$')
    
    
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    if file_name is not None:
        plt.savefig(file_name)
        plt.clf()
    else:
        plt.show()


def kNN(k_val, xValsTrain, xValsTest, yValsTrain, yValsTest):
    """Run kNN regression

    Args:
        kVal (int): Value to use as k for kNN
        xValsTrain (np.array): 2-d np.array holding the photometry used for training
        xValsTest (np.array): 2-d np.array holding the photometry used for testing
        yValsTrain (np.array): 1-d np.array holding the measured redshift for training
        yValsTest (np.array): 1-d np.array holding the measured redshift for testing

    Returns:
        np.array: 1-d np.array holding the predictions
        float: R^2 Coefficient of Determination. 
    """
    neigh = KNeighborsRegressor(n_neighbors=k_val, metric = "mahalanobis", metric_params={"V":np.cov(xValsTrain.astype(np.float64), rowvar=False)})
    neigh.fit(xValsTrain.astype(np.float64),np.squeeze(yValsTrain).astype(np.float64))
    predictions = neigh.predict(xValsTest.astype(np.float64))
        
    return predictions.ravel(), neigh.score(xValsTest,np.squeeze(yValsTest)), neigh

def kNN_cross_val(k_val, k_fold_val, random_seed, x_vals_train, y_vals_train, tqdm_disable = False):
    """Run kNN regression

    Args:
        k_Val (int): Value to use as k for kNN
        k_fold_val (int): How many folds in cross-validation to complete
        random_seed (int): Random seed to use in the cross-validation split. 
        x_vals_train (np.array): 2-d np.array holding the photometry used for training
        x_vals_test (np.array): 2-d np.array holding the photometry used for testing
        y_vals_train (np.array): 1-d np.array holding the measured redshift for training
        y_vals_test (np.array): 1-d np.array holding the measured redshift for testing

    Returns:
        np.array: 1-d np.array holding the predictions
        float: R^2 Coefficient of Determination. 
    """
    out_rate = []
    k_fold = KFold(n_splits=k_fold_val, random_state=random_seed, shuffle=True)
    for train_index, test_index in tqdm(k_fold.split(x_vals_train), total=k_fold_val, disable=tqdm_disable, leave=None):
        #Set up datasets - x vals will be shared between regression and classification
        x_vals_cross_train = x_vals_train[train_index]
        x_vals_cross_test = x_vals_train[test_index]
        # y vals for regression tests
        y_vals_cross_train = y_vals_train[train_index]
        y_vals_cross_test = y_vals_train[test_index]

        x_vals_cross_train, x_vals_cross_test, _, _ = norm_x_vals(x_vals_cross_train, x_vals_cross_test)

        neigh = KNeighborsRegressor(n_neighbors=k_val, metric = "mahalanobis", metric_params={"V":np.cov(x_vals_cross_train.astype(np.float64), rowvar=False)})
        neigh.fit(x_vals_cross_train.astype(np.float64),np.squeeze(y_vals_cross_train).astype(np.float64))
        predictions = neigh.predict(x_vals_cross_test.astype(np.float64))
        out_rate.append(outlier_rate(norm_residual(y_vals_cross_test, predictions)))
        
    return np.mean(out_rate)


def kNN_class(kVal, xValsTrain, xValsTest, yValsTrain, yValsTest):
    """Run kNN classification. 

    Args:
        kVal (int): Value to use as k for kNN
        xValsTrain (np.array): 2-d np.array holding the photometry used for training
        xValsTest (np.array): 2-d np.array holding the photometry used for testing
        yValsTrain (np.array): 1-d np.array holding the measured redshift for training
        yValsTest (np.array): 1-d np.array holding the measured redshift for testing
        distType ([type]): Integer used to determine the distance metric. If less than 5, minkowski distance used. If more, mahalanobis. 

    Returns:
        np.array: 1-d np.array holding the predictions
        float: accuracy of the predictions
    """
    neigh = KNeighborsClassifier(n_neighbors = kVal, metric = "mahalanobis", metric_params={"V":np.cov(xValsTrain.astype(np.float64), rowvar=False)})
    neigh.fit(xValsTrain.astype(np.float64),np.squeeze(yValsTrain).astype(str))
    predictions = neigh.predict(xValsTest.astype(np.float64))
    return predictions.astype(np.float64).ravel(), neigh.score(xValsTest.astype(np.float64),np.squeeze(yValsTest).astype(str)).astype(np.float64), neigh


def kNN_class_cross_val(k_val, k_fold_val, random_seed, x_vals_train, y_vals_train, num_bins, tqdm_disable = False):
    """Run kNN classification. 

    Args:
        kVal (int): Value to use as k for kNN
        xValsTrain (np.array): 2-d np.array holding the photometry used for training
        xValsTest (np.array): 2-d np.array holding the photometry used for testing
        yValsTrain (np.array): 1-d np.array holding the measured redshift for training
        yValsTest (np.array): 1-d np.array holding the measured redshift for testing
        distType ([type]): Integer used to determine the distance metric. If less than 5, minkowski distance used. If more, mahalanobis. 

    Returns:
        np.array: 1-d np.array holding the predictions
        float: accuracy of the predictions
    """
    out_rate = []
    k_fold = KFold(n_splits=k_fold_val, random_state=random_seed, shuffle=True)
    for train_index, test_index in tqdm(k_fold.split(x_vals_train), total=k_fold_val, disable=tqdm_disable, leave=None):
        #Set up datasets - x vals will be shared between regression and classification
        x_vals_cross_train = x_vals_train[train_index]
        x_vals_cross_test = x_vals_train[test_index]
        # y vals for regression tests
        y_vals_cross_train = y_vals_train[train_index]
        y_vals_cross_test = y_vals_train[test_index]
        y_vals_train_bin, y_vals_test_bin, _, _\
            = bin_data_func(np.copy(y_vals_cross_train), np.copy(y_vals_cross_test), num_bins)
        x_vals_cross_train, x_vals_cross_test, _, _ = norm_x_vals(x_vals_cross_train, x_vals_cross_test)
        neigh = KNeighborsClassifier(n_neighbors=k_val, metric = "mahalanobis", metric_params={"V":np.cov(x_vals_cross_train.astype(np.float64), rowvar=False)})
        neigh.fit(x_vals_cross_train.astype(np.float64),np.squeeze(y_vals_train_bin).astype(str))
        predictions = neigh.predict(x_vals_cross_test.astype(np.float64))
        norm_resid = norm_residual(y_vals_test_bin.astype(np.float64), predictions.astype(np.float64))
        out_rate.append(outlier_rate(norm_resid))

        
    return np.mean(out_rate)


def rf(treeVal, xValsTrain, xValsTest, yValsTrain, yValsTest, randomState=None):
    """Run random forest regression

    Args:
        treeVal (Integer): Number of trees to use in classififcation
        xValsTrain (np.array): 2-d numpy array holding the training photometry
        xValsTest (np.array): 2-d numpy array holding the testing photometry
        yValsTrain (np.array): 1-d numpy array holding the training redshifts
        yValsTest (np.array): 1-d numpy array holding the test redshifts
        randomState (bool or integer, optional): If nothing given, defaults to 42. Otherwise, sets the random state. Defaults to False.

    Returns:
        np.array: 1-d np.array holding the predictions
        float: R^2 Coefficient of Determination. 
    """
    if randomState is None:
        randomState = 42
    rf_model = RandomForestRegressor(treeVal, random_state=randomState, n_jobs=-1)
    rf_model.fit(xValsTrain.astype(np.float64),np.squeeze(yValsTrain).astype(np.float64))
    predictions = rf_model.predict(xValsTest.astype(np.float64))
    return predictions.ravel(), rf_model.score(xValsTest.astype(np.float64),np.squeeze(yValsTest).astype(np.float64)), rf_model


def rf_cross_val(tree_val, k_fold_val, x_vals_train, y_vals_train, random_state=42, tqdm_disable=False):
    """Run random forest regression

    Args:
        treeVal (Integer): Number of trees to use in classififcation
        k_fold_val (Integer): Number of folds to use. 
        xValsTrain (np.array): 2-d numpy array holding the training photometry
        xValsTest (np.array): 2-d numpy array holding the testing photometry
        yValsTrain (np.array): 1-d numpy array holding the training redshifts
        yValsTest (np.array): 1-d numpy array holding the test redshifts
        randomState (bool or integer, optional): If nothing given, defaults to 42. Otherwise, sets the random state. Defaults to False.

    Returns:
        np.array: 1-d np.array holding the predictions
        float: R^2 Coefficient of Determination. 
    """
    out_rate = []
    k_fold = KFold(n_splits=k_fold_val, random_state=random_state, shuffle=True)
    for train_index, test_index in tqdm(k_fold.split(x_vals_train), total=k_fold_val, disable=tqdm_disable, leave=None):
        #Set up datasets - x vals will be shared between regression and classification
        x_vals_cross_train = x_vals_train[train_index]
        x_vals_cross_test = x_vals_train[test_index]
        # y vals for regression tests
        y_vals_cross_train = y_vals_train[train_index]
        y_vals_cross_test = y_vals_train[test_index]

        x_vals_cross_train, x_vals_cross_test, _, _ = norm_x_vals(x_vals_cross_train, x_vals_cross_test)

        neigh = RandomForestRegressor(tree_val, random_state=random_state, n_jobs=-1)
        neigh.fit(x_vals_cross_train, np.squeeze(y_vals_cross_train))
        predictions = neigh.predict(x_vals_cross_test.astype(str))
        out_rate.append(outlier_rate(norm_residual(y_vals_cross_test, predictions)))
        
    return np.mean(out_rate)


def rf_class(treeVal, xValsTrain, xValsTest, yValsTrain, yValsTest, num_bins, randomState = None):
    """Run Random Forest Classification

    Args:
        treeVal (Integer): Number of trees to use in classififcation
        xValsTrain (np.array): 2-d numpy array holding the training photometry
        xValsTest (np.array): 2-d numpy array holding the testing photometry
        yValsTrain (np.array): 1-d numpy array holding the training redshifts
        yValsTest (np.array): 1-d numpy array holding the test redshifts
        randomState (bool or integer, optional): If nothing given, defaults to 42. Otherwise, sets the random state. Defaults to False.

    Returns:
        np.array: Predicted redshifts. 
        float: Prediction accuracies
    """
    if randomState is None:
        randomState = 42
    rf_model = RandomForestClassifier(treeVal, random_state=randomState, n_jobs=-1)
    rf_model.fit(xValsTrain.astype(np.float64),np.squeeze(yValsTrain).astype(str))
    predictions = rf_model.predict(xValsTest.astype(np.float64))
    return predictions.ravel(), rf_model.score(xValsTest.astype(np.float64),np.squeeze(yValsTest).astype(str)), rf_model


def rf_class_cross_val(tree_val, k_fold_val, x_vals_train, y_vals_train, num_bins, random_state=42, tqdm_disable=False):
    """Run Random Forest Classification

    Args:
        treeVal (Integer): Number of trees to use in classififcation
        xValsTrain (np.array): 2-d numpy array holding the training photometry
        xValsTest (np.array): 2-d numpy array holding the testing photometry
        yValsTrain (np.array): 1-d numpy array holding the training redshifts
        yValsTest (np.array): 1-d numpy array holding the test redshifts
        randomState (bool or integer, optional): If nothing given, defaults to 42. Otherwise, sets the random state. Defaults to False.

    Returns:
        np.array: Predicted redshifts. 
        float: Prediction accuracies
    """
    out_rate = []
    k_fold = KFold(n_splits=k_fold_val, random_state=random_state, shuffle=True)
    for train_index, test_index in tqdm(k_fold.split(x_vals_train), total=k_fold_val, disable=tqdm_disable, leave=None):
        #Set up datasets - x vals will be shared between regression and classification
        x_vals_cross_train = x_vals_train[train_index]
        x_vals_cross_test = x_vals_train[test_index]
        # y vals for regression tests
        y_vals_cross_train = y_vals_train[train_index]
        y_vals_cross_test = y_vals_train[test_index]
        
        y_vals_train_bin, y_vals_test_bin, _, _\
            = bin_data_func(y_vals_cross_train, y_vals_cross_test, num_bins)

        x_vals_cross_train, x_vals_cross_test, _, _ = norm_x_vals(x_vals_cross_train, x_vals_cross_test)

        neigh = RandomForestClassifier(tree_val, random_state=random_state, n_jobs=-1)
        neigh.fit(x_vals_cross_train,np.squeeze(y_vals_train_bin.astype(str)))
        predictions = neigh.predict(x_vals_cross_test)
        out_rate.append(outlier_rate(norm_residual(y_vals_test_bin.astype(np.float64), predictions.astype(np.float64))))
        
    return np.mean(out_rate)

def bin_data_func(train_redshift, test_redshift, numBins, bin_edges = None, bin_medians = None):
    """Function to bin the data

    Args:
        redshiftVector (np.array): 1-d numpy array holding the redshifts to be binned
        numBins (integer): Number of bins to use
        maxRedshift (float, optional): Value to use as the highest bin edge. Defaults to 1.5.

    Returns:
        np.array: List containing the binned redshifts
        np.array: List containing each of the bin edges
        np.array: List containing the centres of the bins
    """
    if bin_edges is None:
        sorted_training = np.sort(train_redshift, axis=None)
        numPerBin = sorted_training.shape[0]//numBins #Integer division!
        bin_edges = [0]
        # Find each of the bin edges
        for i in range(1, numBins):
            bin_edges.append(i * numPerBin)
        bin_edges.append(sorted_training.shape[0]-1)
        # Replace the indices of the bin edges with the bin edge values
        bin_edges = sorted_training[bin_edges]
        bin_edges[-1] = np.median(sorted_training[np.where(sorted_training > bin_edges[-2])])
        # New list to hold the median of each bins
        bin_medians = []
        for i in range(1, numBins + 1):
            if i < numBins:
                bin_medians.append(np.median([bin_edges[i-1], bin_edges[i]]))
            else:
                bin_medians.append(np.median(train_redshift[np.where((train_redshift >= bin_edges[i-1]) &\
                    (train_redshift < np.max(train_redshift)))[0]])) 
        # Bin the data
    for i in range(1, numBins + 1):
        if i < numBins:
            if i == 1:
                train_redshift[np.where((train_redshift < bin_edges[i]))[0]] = bin_medians[i-1]
                test_redshift[np.where((test_redshift < bin_edges[i]))[0]] = bin_medians[i-1]
            else:
                train_redshift[np.where((train_redshift >= bin_edges[i-1]) &\
                     (train_redshift < bin_edges[i]))[0]] = bin_medians[i-1]
                test_redshift[np.where((test_redshift >= bin_edges[i-1]) &\
                     (test_redshift < bin_edges[i]))[0]] = bin_medians[i-1]
        else: 
            train_redshift[np.where((train_redshift >= bin_edges[i-1]))[0]] = bin_medians[i-1]
            test_redshift[np.where((test_redshift >= bin_edges[i-1]))[0]] = bin_medians[i-1]
    return train_redshift, test_redshift, bin_edges, bin_medians


def norm_x_vals(x_vals_train, x_vals_test):
    """Normalise x_vals, based on the training sample..

    Args:
        x_vals_train (np.array): x_vals training sample
        x_vals_test (np.array): x_vals test sample

    Returns:
        np.array: Normalised x_vals_train
        np.array: Normalised y_vals_train
        np.array: 1-d array with the mean of each feature
        np.array: 1-d array with the std dev of each feature
        
    """
    mean_norm = np.mean(x_vals_train, axis = 0)
    std_norm = np.std(x_vals_train, axis = 0)
    x_vals_train = (x_vals_train - mean_norm) / std_norm
    x_vals_test = (x_vals_test - mean_norm) / std_norm

    return x_vals_train, x_vals_test, mean_norm, std_norm

