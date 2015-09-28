# primal_dual_scca

Implementation of the Primal-Dual Sparse Canonical Correlation Analysis algorithm described in: 
Hardoon, David R., and John Shawe-Taylor. "Sparse canonical correlation analysis." Machine Learning 83.3 (2011): 331-353.

Implementation chiefly due to Kristian Nybo, portions of code due to Viivi Uurtio, Juho Rousu and David R. Hardoon.

In order to run the scca_cvx_singleprog_tau function the cvx toolbox needs to be installed. Input your data to the SCCAwrapper_cvx function and the canonical weights will be given in the output. 
