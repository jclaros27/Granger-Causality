# Granger-Causality
Bivariate Granger Causality in Python

We want to know if the past of a variable $Y$ make us predict what happend in the variable $X$ better than considering only the past of $Y$. If this is true, we can affirm that there is a granger casual betwen $Y$ and $X$. 

Given the two time series, $X(t)$ and $Y(t)$, it is possible to construct an auto-regressive (AR) model for each one in which each time point is the result of a linear combination of their $p$ past points plus a residual (assumed white noise):

$$ X(t) = \displaystyle\sum_{k=1}^p a_{1,k}X(t-k) + \varepsilon_1, $$

$$ Y(t) = \displaystyle\sum_{k=1}^p a_{2,k}Y(t-k) + \varepsilon_2, $$

where the matrix $a_{i,j}$ contains the weights of the contributions of the past of $X(t)$ and $Y(t)$, and $\varepsilon_i$ are the residuals. The order of the AR model $p$ can be estimated by different criteria such as Akaike Information Criterion (AIC) or the Bayesian Information Criterion (BIC). In this repository $p$ is obtained as the minimum value of those obtained by this two criteria. 

Modelling the past-influence of the other series is what is called Bivariate Auto-regressive model, given rise to:

$$ X(t) = \displaystyle\sum_{k=1}^p a_{11,k}X(t-k) + \displaystyle\sum_{k=1}^p a_{12,k}Y(t-k) \varepsilon_{12}, $$
$$ Y(t) = \displaystyle\sum_{k=1}^p a_{21,k}X(t-k) + \displaystyle\sum_{k=1}^p b_{22,k}Y(t-k) \varepsilon_{21}. $$

If the variance of the error of the bivariate model $\sigma^2_{\varepsilon_{12}}$ is smaller than the one of the auto-regressive one $\sigma^2_{\varepsilon_1}$, this would mean that the past of $Y(t)$ improves the prediction of $X(t)$, then, we can affirm that $Y(t)$ Granger-causes $X(T)$. This causality can be quantified as follows:

$$ GC = \log \frac{\sigma_{\varepsilon_{1}}{\sigma_{\varepsilon_{12}}}}).$$

Of course, Granger Causality can be expanded to more variables using multivariate AR models, but this is not (yet) implemented in this repository.
