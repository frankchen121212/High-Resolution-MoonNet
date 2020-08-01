import numpy as np
from utils.template_match_target import template_match_t2c
from utils.processing import get_id
########################
def evaluation(data, craters, dim, model, beta=1):
    """Function that prints pertinent metrics at the end of each epoch.

    Parameters
    ----------
    data : hdf5
        Input images.
    craters : hdf5
        Pandas arrays of human-counted crater data.
    dim : int
        Dimension of input images (assumes square).
    model : keras model object
        Keras model
    beta : int, optional
        Beta value when calculating F-beta score. Defaults to 1.
    """
    X, Y = data[0], data[1]

    # Get csvs of human-counted craters
    csvs = []
    minrad, maxrad, cutrad, n_csvs = 3, 50, 0.8, len(X)
    diam = 'Diameter (pix)'
    for i in range(n_csvs):
        csv = craters[get_id(i)]
        # remove small/large/half craters
        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
        csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]
        if len(csv) < 3:    # Exclude csvs with few craters
            csvs.append([-1])
        else:
            csv_coords = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T
            csvs.append(csv_coords)

    # Calculate custom metrics
    recall, precision, fscore = [], [], []
    frac_new, frac_new2, maxrad = [], [], []
    err_lo, err_la, err_r = [], [], []
    frac_duplicates = []
    preds = model.predict(X)
    for i in range(n_csvs):
        if len(csvs[i]) < 3:
            continue
        (N_match, N_csv, N_detect, maxr,
         elo, ela, er, frac_dupes) = template_match_t2c(preds[i], csvs[i],
                                                            rmv_oor_csvs=0)
        if N_match > 0:
            p = float(N_match) / float(N_match + (N_detect - N_match))
            r = float(N_match) / float(N_csv)
            f = (1 + beta**2) * (r * p) / (p * beta**2 + r)
            diff = float(N_detect - N_match)
            fn = diff / (float(N_detect) + diff)
            fn2 = diff / (float(N_csv) + diff)
            recall.append(r)
            precision.append(p)
            fscore.append(f)
            frac_new.append(fn)
            frac_new2.append(fn2)
            maxrad.append(maxr)
            err_lo.append(elo)
            err_la.append(ela)
            err_r.append(er)
            frac_duplicates.append(frac_dupes)
        # else:
            # print("skipping iteration %d,N_csv=%d,N_detect=%d,N_match=%d" %
            #       (i, N_csv, N_detect, N_match))

    return np.mean(recall), np.mean(precision), np.mean(fscore)
    # print("binary XE score = %f" % model.evaluate(X, Y))
    # if len(recall) > 3:
    #     print("mean and std of N_match/N_csv (recall) = %f, %f" %
    #           (np.mean(recall), np.std(recall)))
    #     print("""mean and std of N_match/(N_match + (N_detect-N_match))
    #           (precision) = %f, %f""" % (np.mean(precision), np.std(precision)))
    #     print("mean and std of F_%d score = %f, %f" %
    #           (beta, np.mean(fscore), np.std(fscore)))
    #     print("""mean and std of (N_detect - N_match)/N_detect (fraction
    #           of craters that are new) = %f, %f""" %
    #           (np.mean(frac_new), np.std(frac_new)))
    #     print("""mean and std of (N_detect - N_match)/N_csv (fraction of
    #           "craters that are new, 2) = %f, %f""" %
    #           (np.mean(frac_new2), np.std(frac_new2)))
    #     print("median and IQR fractional longitude diff = %f, 25:%f, 75:%f" %
    #           (np.median(err_lo), np.percentile(err_lo, 25),
    #            np.percentile(err_lo, 75)))
    #     print("median and IQR fractional latitude diff = %f, 25:%f, 75:%f" %
    #           (np.median(err_la), np.percentile(err_la, 25),
    #            np.percentile(err_la, 75)))
    #     print("median and IQR fractional radius diff = %f, 25:%f, 75:%f" %
    #           (np.median(err_r), np.percentile(err_r, 25),
    #            np.percentile(err_r, 75)))
    #     print("mean and std of frac_duplicates: %f, %f" %
    #           (np.mean(frac_duplicates), np.std(frac_duplicates)))
    #     print("""mean and std of maximum detected pixel radius in an image =
    #           %f, %f""" % (np.mean(maxrad), np.std(maxrad)))
    #     print("""absolute maximum detected pixel radius over all images =
    #           %f""" % np.max(maxrad))
    #     print("")

