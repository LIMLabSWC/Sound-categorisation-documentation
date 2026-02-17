# BE Variations

There are 3 variations of BE model. You can find code, psychometrics of parameter sweep, and the notebook for the parameter sweep here :

3 variations:

1. **BE**: 

In this variation, eta_relax is removed. (relaxation is done implicitely by shifting the whole curve upwards when there are negative values, so the slope between two consecutive point decreases because of renormalisation and shrinking of their vertical distance). And the more the slope value decreases, the closer distribution gets to uniform distribution.  

2. **BE_V2**:

In this variation, instead of negating sigmoid, we add sigmoid and calculate the integral analytically. The Parameter sweep is done over different ranges and even logarithmically.

3. **BE_V3**:

In this variation, instead of shifting the curve upwards when it gets negative values, we clip it, so all negative values become zero.

## Conclusion

Remove the $\eta_{relax}$, a.k.a use BE code here, do not use other variations.
