Soundcat hybrid model

Soundcat hybrid model

### Overview

The hybrid model contains a BE model component and an SC model component, and interpolates between them using a model combination strategy. Each component computes a belief about the category, given the stimulus. The hybrid model uses a convex combination of these beliefs to select actions.

### Behavior

The hybrid model contains an initial sensory processing stage, where sensory noise and repulsion are applied to the incoming stimulus as in the original BE/SC models. Sensory processing is shared between the BE and SC components; a single noisy/repulsed sensory representation is used by both components, rather than generating a separate representation for each.

Let $\theta_t = (\theta^{BE}_t, \theta^{SC}_t)$ be the agent-learned parameters of each component model on each trial $t$ (e.g. governing the BE model's boundary location and the SC model's stimulus distributions). Let $\phi$ contain fittable parameters we seek to recover as experimenters (e.g. sensory noise and repulsion parameters, learning rates, etc.).

Behavior is generated as follows. On each trial $t$, noise and repulsion are applied to stimulus $s_t$ to generate sensory representation $x_t \sim p(x_t \mid s_t; \phi)$ as described above. Each component model computes a conditional distribution over the category $c_t$, given the sensory representation. The hybrid model forms a belief about the category as a convex combination of these component distributions:

$$\begin{array}{cccc}
p_{hybrid}(c_t \mid x_t; \theta_t, \phi) & =
& (1-\alpha) & p_{BE}(c_t \mid x_t; \theta^{BE}_t, \phi)
\\
&& + \alpha & p_{SC}(c_t \mid x_t; \theta^{SC}_t, \phi)
\end{array}$$

$\alpha \in [0, 1]$ is a mixing weight that controls the contribution of each component model. At the endpoints, $\alpha=0$ corresponds to pure BE behavior and $\alpha=1$ corresponds to pure SC behavior. $\alpha$ is a fittable parameter, and does not change over trials.

The internal choice $\hat{c}_t$ (i.e. intended response) is chosen as the most probable category ($A$ or $B$) according to the conditional distribution above. This maximizes expected reward under the 0-1 loss/symmetric rewards used in our experiments:

$$\hat{c}_t = \underset{c_t \in \{A, B\}}{\arg \max}
\ p_{hybrid}(c_t \mid x_t; \theta_t, \phi)$$

The action $a_t$ (final behavioral output) is influenced by decision noise, and is obtained by randomly keeping or flipping the internal choice to the opposite category. Lapse rates $\lambda_A$ and $\lambda_B$ give the probability of flipping a category $A$ or $B$ choice, respectively:

$$\begin{gather*}
p(a_t=B \mid \hat{c}_t=A) = \lambda_A
\\
p(a_t=A \mid \hat{c}_t=B) = \lambda_B
\end{gather*}$$

Reward $r_t$ is received depending on the chosen action and true category. Following this, agent-learned parameters are updated for the next trial. Parameters for each component model are updated using the same learning rules $L_{BE}$ and $L_{SC}$ used by our current BE and SC models (depending on the sensory input, action, and reward):

$$\theta^{BE}_{t+1} \gets L_{BE}(\theta^{BE}_t, x_t, a_t, r_t; \phi)$$

$$\theta^{SC}_{t+1} \gets L_{SC}(\theta^{SC}_t, x_t, a_t, r_t; \phi)$$

### Fitting

Fittable parameters $\phi$ include:

- Sensory parameters (controlling noise, repulsion)
- Fittable BE/SC component parameters (e.g. kernel widths for SC model)
- Model combination weight $\alpha$
- Lapse rates $\lambda_A, \lambda_B$
- Parameters for BE/SC learning rules (e.g. learning rates)

For the current paper, the loss function is computed identically to the original BE/SC models. For a given choice of $\phi$, the model is run through simulated trials. Sampled actions are used to estimate the update or conditional psychometric matrix. The loss is computed as the squared error between the model and empirical update or conditional psychometric matrices.

Optimization algorithm TBD. In principle, the hybrid model could be fit using grid search, as currently used for the BE and SC models. However, it contains around twice the number of parameters, as it includes both BE and SC components. So, grid search might be prohibitively expensive. Try Bayesian optimization, CMA-ES, or other black box optimization algorithms for stochastic objective functions.

### Notes

- Decision noise/lapse rates were not included in the original BE and SC models. This should be added to the original models before comparing them to the hybrid model

- Re-running the original BE/SC models with changes is equivalent to running the hybrid model with $\alpha$ pinned to $0$ or $1$, respectively.