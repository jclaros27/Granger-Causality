# Granger-Causality
Bivariate Granger Causality in Python

Granger Causality is a linear measurement widely used in different fields such as Economics or Neuroscience.

About granger casuality. We want to know if the past of a signal B make us predict what happend in the signal A better than considereing the past of B. If this is true, we can affirm that there is a granger casual betwen B and A

Univariate autogressive models. 
model in which the current value of a sinal can be predicted by a weighted linear combination of previous fdata points.
(autorregresive coefficiente)
\begin{equation}
    X_t = \displaystyle\sum_{n=1}^k a_nX_{t-n} + e_t 
    \label{eq:Univariate_autorregresive_model}
\end{equation}

(bivariate autorregresive models
\begin{equation}
    X_t = \displaystyle\sum_{n=1}^k a_nX_{t-n} + \displaystyle\sum_{n=1}^k b_nY_{t-n} + \varespsilon_t
    \label{eq:Bivariate_autorregresive_model}
\end{equation}
)
\begin{equation}
    GC = \log \bigg(\displaystyle\frac{\sigma_e}{\sigma_{\varepsilon}}  \bigg)
    \label{eq:qunatification_granger_causality}
\end{equation}
