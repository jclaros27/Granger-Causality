import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR, VARResults
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

def get_pmin(data,variables_pairs,pmax=51):
    '''
    Compute the lag that minimizes the BIC and AIC criterion for auto-regressive models.

    inputs:
    -------
    data: pandas.DataFrame object containing the time series data to analyze.
    variables_pairs: list of tuples containing the name of the variables and the name 
    of the time series to analyze.
    pmax: int, the maximum number of lags to consider. Default is 51.


    return:
    -------
    pmin: int, the lag that minimizes the BIC and AIC criterion.
    '''

    pseq = np.arange(1,pmax,1)
    aic, bic = [], []
    for i in pseq:
        results = VAR(data[variables_pairs]).fit(i)
    aic.append( results.aic )
    bic.append( results.bic )
    pmin = int( np.min([np.min(bic), np.min(aic)]))
    if pmin == pmax:
        print("Pmax should be increased. Minimum p is not guaranteed")
    return pmin 

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an 
    Augmented Dickey-Fuller Test report. 
    
    Parameters
    ----------
    series: pandas.Series object containing the time series data to analyze.
    title: str, optional, the title of the report. Default is ''.

    Returns
    -------
    stationarity_condition: boolean, True if the series satisfies stationary 
    condition, False if not.

    Notes
    -----
    Source: https://github.com/BessieChen/Python-for-Financial-Analysis-and-Algorithmic-Trading/blob/master/08-Time-Series-Analysis/4-ARIMA-and-Seasonal-ARIMA.ipynb
    """
    
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna()) # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    for key,val in result[4].items():
        out[f'critical value ({key})']=val
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
        stationarity_condition = True
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        stationarity_condition = False

    return stationarity_condition

def granger_causality_ratio(data,variables):
    '''
    Compute the bivariate Granger Causality rato between all posible pairs
    of the variables given. 
    First, the stationarity condition is check. If is not satisfied Granger Causality 
    should not be applied but still it is computed.

    Parameters
    ----------
    data: pandas.DataFrame object containing the time series data to analyze.
    variables: list, the name of the time series to analyze.

    Returns
    -------
    df_ratio: pandas.DataFrame object containing the Granger Causality ratio.
    df_results: pandas.DataFrame object containing  the results of the Granger Causality test:
        (0) X: endogenous variable
        (1) Y: exogenous variable
        (2) Error std: standard deviation of the error of the fitting
        (3) Correlation: pearson correlation coefficient between model fit and real data
        (4) GC p-value: p-value of the Granger Causality test (F test)
        (5) GC ratio: GC ratio of the Granger Causality
    '''

    df_ratio = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    variables_pairs = []    
    print("Stationarity of each time series")
    print("--------------------------------")
    stat_cond = {}
    for i, r in enumerate(variables):
        stat_cond[r] = adf_test(data[r],title=r)
        print(' ')
    
    for i,r in enumerate(variables):
        for j,c in enumerate(variables[i+1:]):
            variables_pairs.append([r,c])
    columns = ["X","Y","Error std","Correlation","GC p-value","GC ratio"]
    ncolumns = len(columns)
    ncombinations = 4*len(variables_pairs)
    df_results = pd.DataFrame(np.zeros((ncombinations,ncolumns)).astype(str), columns=columns)
    print(' ')
    
    pmin_ = []
    for k, pair in enumerate(variables_pairs):
        pmin_.append( get_pmin(data[pair],pair) )
    pmin = np.max(pmin_)
    print(f"Number coeficients for the Autorregresive models {pmin}")

    for k, (r,c) in enumerate(variables_pairs):
        df_results.iloc[4*k]["X"] = r
        df_results.iloc[4*k]["Y"] = "-"
        df_results.iloc[4*k+1]["X"] = r
        df_results.iloc[4*k+1]["Y"] = c
        df_results.iloc[4*k+2]["X"] = c
        df_results.iloc[4*k+2]["Y"] = "-"
        df_results.iloc[4*k+3]["X"] = c
        df_results.iloc[4*k+3]["Y"] = r
        
        # vector autorregressive model
        var_model = VAR(data[[r,c]]).fit(pmin)
        index = var_model.fittedvalues.index

        # predctions of the VAR model
        r_std_pred = var_model.fittedvalues[r].values 
        c_std_pred = var_model.fittedvalues[c].values

        # real data: must use index to have same dimension
        r_data = data[r].loc[index].values
        c_data = data[c].loc[index].values

        # error of the fitted values
        r_std_error = r_std_pred-r_data 
        c_std_error = c_std_pred-c_data

        # corelation between data and model
        df_results.iloc[4*k+1]["Correlation"] = np.corrcoef(r_std_pred, r_data)[0,1]
        df_results.iloc[4*k+3]["Correlation"] = np.corrcoef(c_std_pred, c_data)[0,1]

        # error std of the fitted values
        df_results.iloc[4*k+1]["Error std"] = np.std(np.abs(r_std_error))
        df_results.iloc[4*k+3]["Error std"] = np.std(np.abs(c_std_error))
        
        # p-value of the GC (F test)
        df_results.iloc[4*k+1]["GC p-value"] = var_model.test_causality(r,c).pvalue
        df_results.iloc[4*k+3]["GC p-value"] = var_model.test_causality(c,r).pvalue

        # autoregression model of each variable
        autoreg_model_r = AutoReg(data[r],lags=pmin).fit()
        r_auto_pred     = autoreg_model_r.fittedvalues.loc[index].values
        autoreg_model_c = AutoReg(data[c],lags=pmin).fit() 
        c_auto_pred     = autoreg_model_c.fittedvalues.loc[index].values

        # error of the fitted values
        r_auto_error = r_auto_pred-r_data
        c_auto_error = c_auto_pred-c_data

        # correlation between data and model
        df_results.iloc[4*k]["Correlation"] = np.corrcoef(r_auto_pred,r_data)[0,1]
        df_results.iloc[4*k+2]["Correlation"] = np.corrcoef(c_auto_pred,c_data)[0,1]

        # error std f the fitted values
        df_results.iloc[4*k]["Error std"] = np.std(np.abs(r_auto_error))
        df_results.iloc[4*k+2]["Error std"] = np.std(np.abs(c_auto_error))
        
        # p-value of the GC (F test)
        df_results.iloc[4*k]["GC p-value"] = var_model.test_causality(r,r).pvalue
        df_results.iloc[4*k+2]["GC p-value"] = var_model.test_causality(c,c).pvalue    
        
        # GC ratio
        df_results.iloc[4*k+1]["GC ratio"] = np.log(df_results.iloc[4*k]["Error std"]/df_results.loc[4*k+1]["Error std"])
        df_results.iloc[4*k+3]["GC ratio"] = np.log(df_results.iloc[4*k+2]["Error std"]/df_results.iloc[4*k+3]["Error std"])
        
        # filtering the ratio according p-value
        if df_results.loc[4*k+1]["GC p-value"] > 0.05:
            df_ratio.loc[c,r] = 0.0 
        else:
            df_ratio.loc[c,r] = df_results.iloc[4*k+1]["GC ratio"]
        
        if df_results.loc[4*k+3]["GC p-value"] > 0.05:
            df_ratio.loc[r,c] = 0.0
        else:
            df_ratio.loc[r,c] = df_results.iloc[4*k+3]["GC ratio"]

    df_results = df_results.drop_duplicates().sort_values("X").style.hide_index()
    print(' ')
    print("Granger Causality results")
    display(df_results)
    print("Granger Causality ratio, p-value filtered")
    display(df_ratio)
    
    return df_results, df_ratio, stat_cond