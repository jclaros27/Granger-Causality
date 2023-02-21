# Granger-Causality
Bivariate Granger Causality in Python

We want to know if the past of a variable $Y$ make us predict what happend in the variable $X$ better than considering only the past of $X$. If this is true, we can affirm that there is a granger casual betwen $Y$ and $X$. 

Given the two time series, $X(t)$ and $Y(t)$, it is possible to construct an Auto-Regressive (AR) model for each one in which each time point is the result of a linear combination of their $p$ past points plus a residual (assumed white noise):

$$ X(t) = \displaystyle\sum_{k=1}^p a_{1,k}X(t-k) + \varepsilon_1(t), $$

$$ Y(t) = \displaystyle\sum_{k=1}^p a_{2,k}Y(t-k) + \varepsilon_2(t), $$

where the matrix $a_{i,j}$ contains the weights of the contributions of the past of $X(t)$ and $Y(t)$, and $\varepsilon_i$ are the residuals. The order of the AR model $p$ can be estimated by different criteria such as Akaike Information Criterion (AIC) or the Bayesian Information Criterion (BIC). In this repository $p$ is obtained as the minimum value of those obtained by these two criteria. 

Modelling the past-influence of the other series is what is called Bivariate Auto-Regressive model, given rise to:

$$ X(t) = \displaystyle\sum_{k=1}^p a_{11,k}X(t-k) + \displaystyle\sum_{k=1}^p a_{12,k}Y(t-k) + \varepsilon_{12}(t), $$

$$ Y(t) = \displaystyle\sum_{k=1}^p a_{21,k}X(t-k) + \displaystyle\sum_{k=1}^p b_{22,k}Y(t-k) + \varepsilon_{21}(t). $$

If the variance of the error of the bivariate model $\sigma^2_{\varepsilon_{12}}$ is smaller than the one of the auto-regressive one $\sigma^2_{\varepsilon_1}$, this would mean that the past of $Y(t)$ improves the prediction of $X(t)$. Then, we can affirm that $Y(t)$ Granger-causes $X(t)$ and this interaction can be quantified as follows:

$$ GC = \log \displaystyle\frac{ \sigma_{\varepsilon_{1}} }{ \sigma_{\varepsilon_{12}} }.$$

Of course, Granger Causality can be expanded to more variables using multivariate AR models, but this is not (yet) implemented in this repository.

For deeper information of the functions used, you can check out the following links:

(i) Vector Auto-Regressive model (VAR): https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.var_model.VAR.html

(ii) Auto-Regressive model (AR): https://www.statsmodels.org/dev/generated/statsmodels.tsa.ar_model.AutoReg.html

(iii) Augmented Dickey-Fuller (ADF) test: ttps://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html 
This is used to check if the time series of interest satifies the condition of stationarity.
