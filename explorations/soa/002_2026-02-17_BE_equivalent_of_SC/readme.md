# BE equivalent of SC

As we know that

$$P(B|s) = \frac{P(s|B) P(B)}{P(s)} = \frac{P(s|B) P(B)}{P(s|A)P(A)+P(s|B)P(B)}$$

In SC model we have $P(s|B)$ and $P(s|A)$, so we can compute $P(B|s)$. Now we assume that $P(B|s)$ is a monotone non-decreasing function, so we can dervive a BE equivalent of this SC model. Since for BE model with boundary distribution of $f(x)$ we know:

$$P(B|s) = \int_{-\infty}^{s}f(x) \ dx \qquad,$$

So we are able to derive $f(x)$, which means a BE equivalent for our SC model at each trial.

## Conclusion

Since $P(B|s)$ is not a monotone non-decreasing function, it can not be CDF of any function (like $f(x)$ ), because it gives $f(x)<0$ values. So we can not extract a BE equivalent for SC model, so SC model's behaviour can not be replicated with a BE model.
