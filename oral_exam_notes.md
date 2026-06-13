# Oral exam — speaking notes for `oral_exam_slides.pdf`

**Purpose.** A reservoir of material per slide, written for *you* to study from
and to read from on the day. Every term and formula on the slides will be
questioned; this file contains the deep version so you can answer.

**Order of the talk** (matches the slides):

1. Terminology / MBSE solvers — FCI, HF, CCSD, DMC, VMC, NQS.
2. Our ansatz (Slater · cusp · neural correlator · CTNN backflow).
3. CTNN architecture.
4. Paper A — four questions: Q1 generalization, Q2 expressivity (depth/width),
   Q3 curse of dimensionality (Barron + intrinsic d), Q4 optimization.
5. Paper B — Q1–Q3 in one slide; Q4 in depth; catch-22 result.
6. Results: SR+VMC accuracy; PINN/collocation results; tricks; SR↔Adam regime
   boundary.
7. Inputs / attribution.
8. Intrinsic dimensionality of the learned manifold.
9. Three-body sensitivity.
10. Synthesis.
11. Backups: NTK proof; Barron proof; design choices → preconditioners; cusp
    log-divergence; force-aligned backflow; PINN–CTNN coupling; N=20 J vs BF.

Color scheme: **Paper A in blue**, **Paper B in red**, **our work in green.**

---

## Slide 1 — Title

**On the slide.** Title, subtitle, name, date.

**Opening line.** *"I was asked to read two articles. Paper A — Berner, Grohs,
Kutyniok, Petersen — is a comprehensive survey of why neural networks work
mathematically. Paper B — De Ryck and Mishra — is a focused numerical analysis
of physics-informed neural networks. Both apply to my thesis because I built a
PINN for the many-body Schrödinger equation. I will present each paper through
its own internal logic, and use our 2D quantum-dot calculations as the worked
example throughout."*

**Posture.** Examiners are from quantum many-body. Tie every deep-learning
abstraction back to a physics object the moment it appears.

---

## Slide 2 — Outline / running thread

**On the slide.** Numbered list of the four blocks, plus the "manifold
exists + manifold reachable" running thread.

**Material.**

- The MBSE Hilbert space is exponentially large (FCI basis grows
  combinatorially; HF discards correlation). But the *physical* ground state
  is concentrated on a low-dimensional manifold defined by symmetry,
  short-range correlation, and analytic structure (cusp). Neural networks are
  machines for *finding such manifolds without expanding against the full
  Hilbert space*.
- Paper A → *does the manifold exist mathematically?* (Barron, depth,
  compositionality, symmetry).
- Paper B → *is the manifold reachable by gradient descent?* (only if
  $\kappa(L^*L \circ TT^*)$ is controlled).
- Our work fits both: architecture encodes the manifold's symmetry (Paper A
  territory), conditioning lets the optimizer reach it (Paper B territory).

**Empirical hook.** The learned correlator's effective rank is ≤ 3 across the
whole $(N, \omega)$ grid — even for $N{=}20$ in a 40-dimensional configuration
space. The manifold really is there, and we measure its dimension directly.

**Don't forget.** State the thread aloud: *"manifold exists + manifold
reachable."* It is the spine of the whole talk.

---

## Slide 3 — Terminology: the MBSE problem and the standard solvers

**On the slide.** The Hamiltonian, the table of solvers, the green box on
expressivity vs scaling.

**Material — what each method does and where it lives in the
expressivity / scaling plane.**

- **HF (Hartree–Fock).** Mean-field. The wavefunction is a single Slater
  determinant of single-particle orbitals; correlation between particles
  beyond exchange is missed entirely. Scales like $O(N^4)$ (integrals). In
  our problem HF gives a useful starting orbital basis but its absolute
  energies are off by a few percent in the regimes we care about.
- **CCSD (Coupled Cluster Singles & Doubles).** Adds excitations on top of
  HF in a controlled, *truncated* expansion: $|\Psi\rangle = e^T |\Phi_0\rangle$
  with $T = T_1 + T_2$. Scales polynomially ($O(N^6)$ ish). Brilliant in the
  weak/moderate correlation regime; catastrophic in strong-correlation
  regimes — for example our deep Wigner limit, where the reference HF
  determinant is a poor zeroth order, the cluster amplitudes diverge, and
  the method becomes non-variational and non-monotone in basis size. CCSD is
  the canonical example of *trading expressivity for scaling*.
- **FCI (Full Configuration Interaction).** Exact within a finite
  single-particle basis: diagonalize the Hamiltonian on the full
  determinantal subspace. Basis size grows as $\binom{2K}{N}$ for $K$
  spin-orbitals and $N$ electrons — combinatorial explosion in both $N$ and
  $K$. The reference solution where available; impossible for $N \gtrsim
  12$ in any realistic basis.
- **DMC (Diffusion Monte Carlo).** Imaginary-time projection
  $|\Psi_0\rangle \propto \lim_{\tau\to\infty} e^{-\tau \hat H}|\Psi_T\rangle$,
  implemented by branching random walks. Exact for bosons; for fermions it
  carries a *fixed-node* approximation (the nodes are inherited from a trial
  wavefunction $\Psi_T$). The de-facto reference for 2D quantum dots in our
  parameter range — Pederiva, Egger, Umrigar, etc.
- **VMC (Variational Monte Carlo).** Pick a trial $\Psi_\theta$, sample
  $|\Psi_\theta|^2$ by MCMC, evaluate $\langle \hat H \rangle = \mathbb
  E[E_L]$ with $E_L = \hat H \Psi / \Psi$, minimize variationally. The
  upper-bound principle: $\langle \hat H\rangle \ge E_0$ always.
- **NQS (Neural-network Quantum States).** Carleo–Troyer 2017 onward.
  Replace the hand-built trial wavefunction with a neural network and
  optimize inside VMC, typically with Stochastic Reconfiguration. Our work
  is in this lineage.

**Connection.**

- The whole *point* of NQS is to combine the two virtues — expressivity (FCI)
  and scaling (CCSD) — into one object. If a neural ansatz with the right
  inductive bias finds the low-dimensional physical manifold, you escape the
  exponential basis cost (this is exactly Paper A Q3 / Barron) while keeping
  the variational guarantee of VMC.
- The slide's green box is this bet in one sentence: *FCI is expressive but
  not scalable; CCSD is scalable but not expressive; ML is the attempt to
  have both.*

**Don't forget.** When asked which conventional method is the natural
"competitor" for our PINN, say *DMC* — that is the reference standard, and
that is what our numbers in the results table beat or match.

---

## Slide 4 — Our ansatz

**On the slide.** The ansatz formula, the four-piece breakdown, the trap-units
remark, "why the determinant sees backflow but the correlator does not."

**Material.**

- **The ansatz.**
  $$ \Psi_{\theta,\beta}(R) = \mathrm{SD}\bigl(\tilde R + \Delta_\beta(\tilde
  R)\bigr) \, \exp\!\Big(\sum_{i<j} u_{\sigma_i\sigma_j}(\tilde r_{ij}) +
  W_\theta(\tilde R)\Big) $$
  in trap units $\tilde r = \sqrt\omega \, r$ so the harmonic length is 1.
- **Slater determinant.** Antisymmetry — without it the variational principle
  collapses to the bosonic ground state. The orbitals are HF orbitals in the
  trap basis.
- **Analytic cusp $u_{\sigma\sigma'}$.** Imposes the Kato cusp condition
  on $1/r$ exactly (slope $1$ for antiparallel, $1/3$ in 2D for parallel
  same-spin pairs). Why "exactly": the cusp is a feature of $\Psi$, not of
  $|\Psi|^2$; it gives a piece of $E_L$ that is singular as $r\to 0$ unless
  cancelled exactly. We cancel it analytically; what remains for the network
  to represent is smooth.
- **Neural correlator $W_\theta$.** A Jastrow factor in the exponent — a
  symmetric multi-electron correlation function with three Deep-Sets-style
  branches:
  - $\phi$ — particle branch (per-particle MLP + mean pool);
  - $\psi$ — pair branch (per-pair MLP on six *safe* radial features + raw
    $(\Delta x, \Delta y)$ + spin + mean pool);
  - $\rho$ — readout MLP on $[\bar h^\phi \| \bar h^\psi \| \mathbf g]$ with
    two global statistics $\mathbf g$ (mean radius², mean soft-pair log
    distance).
- **CTNN backflow $\Delta_\beta$.** A message-passing GNN with one round of
  passing, $\tanh$-bounded, last layer zero-initialized, so $\Delta_\beta
  \approx 0$ at start.
- **Why the determinant sees backflow but the correlator/cusp does not.**
  If the cusp factor also saw the backflow-displaced coordinates, the
  $\nabla_\theta E_L$ chain would re-route through $\Delta_\beta$ and create
  a feedback loop: cusp residual → backflow gradient → cusp residual …
  Keeping the analytic cusp on the untransformed coordinates breaks this
  loop. Standard in modern fermionic NQS (FermiNet, PauliNet).
- **Why three networks.** Particle-only Deep Sets pools away relative
  geometry — mathematically, it cannot represent any two-body correlation
  function. Haas et al. proved this rigorously. So a pair branch is not
  redundant, it is *mandatory*.
- **Trap units.** $\tilde r = \sqrt\omega \, r$. The harmonic oscillator
  length is $\ell = 1/\sqrt\omega$ in physical units, so $\tilde r$ has
  $\ell = 1$ in every regime. This makes the input range $O(1)$ across three
  decades of $\omega$ and is the cheapest input-side preconditioner (Paper B
  Q4 territory).

**Don't forget.** The cusp is the single most important piece. It is the
reason the local energy is bounded and the network never has to learn
$1/r$. Without it, every Paper B assumption breaks down (slide 11: bounded
integrand for the quadrature bound; slide 12: $k_{\max}$ in $L^*L$).

---

## Slide 5 — CTNN architecture

**On the slide.** The message / aggregate / update equations, the
$\tanh\cdot s_\beta$ bound, the four "why these choices" bullets, the
empirical $\|\Delta x\|/\ell \lesssim 0.10$ number.

**Material.**

- **Message-passing on the particle graph.** Particles are nodes;
  pair interactions are directed edges. One round of MP:
  - $\mathbf m_{ij} = \phi_\beta(\tilde x_i, \tilde x_j, \mathbf r_{ij},
    \tilde r^{\rm soft}_{ij}, (\tilde r^{\rm soft}_{ij})^2) \cdot w^{\rm
    spin}_{ij}$;
  - $\mathbf m_i = \sum_{j \ne i} \mathbf m_{ij}$;
  - $\delta\tilde x_i = \psi_\beta(\tilde x_i \| \mathbf m_i)$;
  - $\Delta_\beta = \tanh(\delta\tilde X) \cdot s_\beta$,
    $s_\beta = \mathrm{softplus}(s^{\rm raw}_\beta) > 0$, last linear layer
    of $\psi_\beta$ zero-initialized.
- **Symmetries.**
  - Permutation invariance (sum over $j$ is the symmetric aggregation; sum
    over all $i$ of any function of $\delta\tilde x_i$ is permutation
    equivariant).
  - Rotation equivariance — vector messages $\mathbf r_{ij}$, no axis-aligned
    feature.
  - Near-identity at initialization — zero-init + tanh + scale gives
    $\Delta_\beta \approx 0$ and the flow starts from the identity. This is
    Paper A §1.6.2: a ResNet is the Euler discretization of an ODE
    $\dot \phi = h(t, \phi)$, and ODE flows are the cure for
    vanishing/exploding gradients.
  - $k = 1$ round of MP: each particle's update sees the *aggregated*
    neighborhood, so the cell output depends on triplets $\{i, j, k\}$ — up
    to three-body correlations. With $k$ rounds you reach $(k+1)$-body.
- **Empirical guarantee.** $\|\Delta x\|/\ell \lesssim 0.10$ in production.
  Conventional BackflowNets can reach $0.95$ in poorly converged regimes.
  The architectural bound is what keeps backflow from running off into a
  region of configuration space where $|\Psi|$ is small and the local
  energy estimator is meaningless.

### Is the CTNN "just a GNN"? — yes, and here is exactly what it is

This is a fair question and the honest answer is: **the CTNN is a graph
neural network.** Specifically a message-passing GNN with one round of
passing. There is no claim that it is a new species of network. What the
name "CTNN" picks out is a particular *parametrization* and a particular
*set of inductive biases* layered on the standard GNN template. Be ready to
say this plainly.

- **The standard GNN part (the "two feature spaces + learned mapping" you
  noticed).** A message-passing GNN keeps two kinds of features and a pair
  of learned maps between them:
  - *node features* — one vector per particle, $\tilde x_i$;
  - *edge features* — one vector per pair, built from $(\tilde x_i,
    \tilde x_j, \mathbf r_{ij}, \ldots)$;
  - a learned *node$\to$edge* map (form a message on each edge) and a
    learned *edge$\to$node* map (aggregate messages back to the node, then
    update it).
  In the code (`CTNNBackflowNet`) these are literally named `rho_v_to_e`
  and `rho_e_to_v`. So yes — two feature spaces and learned transport maps
  between them is *exactly* what a GNN is. You have correctly identified
  the GNN; there is nothing hidden underneath it.
- **What is specific to ours (why we bother naming it).**
  1. *Two heads sharing one template.* The same message/aggregate/update
     structure is instantiated twice: once as the **correlator** $W_\theta$
     (output: a scalar log-amplitude correction) and once as the
     **backflow** $\Delta_\beta$ (output: a vector displacement per
     particle). One GNN motif, two different output spaces.
  2. *Near-identity ODE-flow parametrization (the "continuous-time"
     reading).* The backflow output is $\tilde X \mapsto \tilde X +
     \Delta_\beta$ with the last layer zero-initialized, a $\tanh$ bound,
     and a learnable positive scale $s_\beta$. So at initialization it is
     the identity map, and it grows smoothly during training — one Euler
     step of an ODE flow (the ResNet picture, Paper A §1.6.2). A generic
     GNN readout has no such near-identity / ODE structure; this is the
     piece that keeps $\|\Delta x\|/\ell \lesssim 0.1$.
  3. *Physics-conditioned messages.* Spin-weighted edges
     $w^{\rm spin}_{ij}$, soft-core safe radial features, rotation-
     equivariant vector messages, zero-mean displacement (centre-of-mass
     conservation). A textbook GNN would not bake in any of these.
- **So the one-sentence answer:** *"It is a one-round message-passing GNN —
  copresheaf-style, with explicit node and edge feature spaces and learned
  node$\leftrightarrow$edge maps. The 'CTNN' label just records two
  specifics: I run the same message-passing motif as two heads (a scalar
  correlator and a vector backflow), and I parametrize the backflow as a
  near-identity ODE flow. The value is in the inductive biases, not in
  being a new kind of network."*

**Connection.** Every architectural decision maps to one of Paper A's
themes — Deep Sets for permutation equivariance, pair branch for GNN-style
message passing, backflow as ResNet/ODE flow, cusp/soft-features for
non-smooth structure outside the Barron class. This is where you can lay
the pipe between "I built this for physical reasons" and "and it happens to
realize a piece of Paper A in code".

**Don't forget.** Three networks is the minimum. Mention Haas if asked.
Three rounds (instead of one) would be a thesis-scale project on its own;
we observed empirically that one round saturates the manifold structure we
care about (intrinsic dim slide).

---

## Slide 6 — Paper A: the four questions

**On the slide.** Risk decomposition, classical bounds vacuous statement,
four-Q list.

### Reading the formula on the slide — every symbol

The slide shows
$$ R(f_S) - R(f^\star) = \underbrace{[\,R(f_S) - R(f_{\mathcal F})\,]}
   _{\text{generalization (estimation)}}
   + \underbrace{[\,R(f_{\mathcal F}) - R(f^\star)\,]}_{\text{approximation}}. $$
Here is what each piece is, in plain words.

- $\mathcal F$ — **the fancy F is the *function class*** (hypothesis class):
  the set of *all* functions your model can possibly represent. For a
  neural network of a fixed architecture, $\mathcal F$ is every function
  you get by sweeping the weights. "Bigger $\mathcal F$" = a more flexible
  model.
- $R(f)$ — **the risk of a function $f$**: its true expected error over the
  data distribution, $R(f) = \mathbb E_{(x,y)\sim P}\,\mathcal L(f,x,y)$.
  Lower is better. (Not to be confused with the residual or with $\mathbb
  R$ — here $R(\cdot)$ is "risk of".)
- $f_S$ — **the function you actually end up with**: the one that minimizes
  the *empirical* risk on your finite training sample $S$, $f_S =
  \arg\min_{f\in\mathcal F}\hat R_S(f)$, with $\hat R_S(f)=\frac1n\sum_i
  \mathcal L(f,x_i,y_i)$. It depends on the random sample $S$.
- $f_{\mathcal F}$ — **the best function *inside* the class**: $f_{\mathcal
  F} = \arg\min_{f\in\mathcal F} R(f)$. The best you could do *if you had
  infinite data* but were still stuck with this architecture. (This is the
  thing you may have been reading as "$f'$" — it is not a derivative, it is
  the in-class optimum.)
- $f^\star$ — **the best function over *all* functions** (the Bayes-optimal
  predictor), with no restriction to $\mathcal F$. The ideal you can never
  beat.
- So the two gaps are:
  - **approximation error** $R(f_{\mathcal F}) - R(f^\star)$: how much the
    *class itself* falls short of the ideal — a property of the
    architecture, not of the data.
  - **generalization / estimation error** $R(f_S) - R(f_{\mathcal F})$: how
    much *worse the function you picked from finite data* is than the best
    one in the class — a property of having only $n$ samples.

### Why this *is* the bias–variance tradeoff

- *Approximation error* plays the role of **bias**: a bigger, more flexible
  $\mathcal F$ can sit closer to $f^\star$, so bias goes **down**.
- *Estimation error* plays the role of **variance**: a bigger $\mathcal F$
  is harder to pin down from $n$ samples (more ways to fit noise), so
  variance goes **up**.
- Classical wisdom: there is a sweet-spot size of $\mathcal F$ that
  balances the two — the U-shaped test-error curve. Complexity measures (VC
  dimension, Rademacher, covering numbers) make "variance" precise and all
  require $\dim(\mathcal F)\ll n$.
- **The puzzle that breaks this** ($\S$1.1.3 *Do We Need a New Theory?*).
  Deep nets take $\mathcal F$ so enormous (parameters $\gg$ data) that
  classical theory predicts the variance term should blow up. Instead they
  fit the training set perfectly (interpolating regime) and *still*
  generalize — often *better* than smaller models. Every classical
  complexity-based bound is *vacuous*: it predicts test error $\sim 1$ when
  the measured test error is $\sim 10^{-2}$. That contradiction is the whole
  reason Paper A's Q1 exists.

**Material — the four questions.**

- **The four questions** — exactly the article's scaffolding.
  - Q1 generalization (§1.2)
  - Q2 expressivity / depth (§1.3)
  - Q3 curse of dimensionality (§1.4)
  - Q4 optimization (§1.5)
  - plus §1.6 architectures, §1.7 features, §1.8 natural sciences.

**Connection.** Our problem sits in the curse-of-dimensionality regime
($Nd = 40$ for $N{=}20$), so Q3 is where our work most directly engages.
But all four questions matter: generalization manifests as $\mathrm{Var}
[E_L]$; expressivity is the manifold's existence; optimization is SR.

**Don't forget.** The article is *honest* about how much is unsolved — say
that explicitly so your framing of our work as a "worked example of partial
answers" is appropriate, not arrogant.

---

## Slide 7 — A–Q1: Generalization

**On the slide.** Double descent; the NTK (clearly stated); the NTK$\leftrightarrow$SR
link; "no unified theory" verdict. *(Norm-based bounds and implicit bias
were dropped from this slide — implicit regularization now lives on the
optimization slide, Slide 10.)*

### Double descent (the one phenomenology fact on the slide)

The classical U-curve of test-error-vs-capacity is only the *left half* of
the real curve. Past the *interpolation threshold* — the capacity at which
training loss first hits zero — test error *falls a second time*. The
article (Belkin et al. 2019) uses this as the empirical fact that finally
breaks the classical bias–variance narrative: more capacity past
interpolation keeps helping. No need to say more than this on the slide.

### NTK — the full story, with every symbol pinned down

This is the part you said was unclear. Read it slowly; the whole thing is
one chain-rule step.

- **What the network is.** $f_\theta : \mathbb R^d \to \mathbb R$, a network
  with parameters $\theta \in \mathbb R^p$. Training moves $\theta$; that
  moves the function $f_\theta$.
- **What $u_t$ is.** Define the **residual function**
  $$ u_t(x) \;=\; f_{\theta_t}(x) - y(x). $$
  It is *how wrong the network is at input $x$*, at training-time $t$. The
  subscript $t$ is gradient-descent time (think of training as a continuous
  flow). $y(x)$ is the target. So $u_t$ is a *function of $x$* that shrinks
  toward 0 as training proceeds; "training succeeded" means $u_\infty
  \approx 0$. (For us $y$ is whatever the loss compares against; the point
  is purely the mechanics.)
- **What $\dot\theta$ (theta-dot) is, and how $u_t$ relates to it.**
  $\dot\theta_t$ means $d\theta/dt$ — the velocity of the parameters under
  gradient flow:
  $$ \dot\theta_t = -\nabla_\theta L(\theta_t). $$
  Now $u_t$ depends on $x$ *and* on $\theta_t$. As $\theta$ moves, $u_t$
  moves. Chain rule (this is the only computation):
  $$ \dot u_t(x) = \underbrace{\nabla_\theta f_{\theta_t}(x)}
     _{\text{how }f\text{ at }x\text{ changes with }\theta}
     \cdot \underbrace{\dot\theta_t}_{\text{how }\theta\text{ moves}}
     = -\sum_{x'} \Theta_{\theta_t}(x,x')\,u_t(x'). $$
  In words: *"how fast the error at $x$ shrinks" = "the network's
  sensitivity at $x$" dotted with "the parameter velocity", and the
  parameter velocity is itself driven by the error at every other point
  $x'$, weighted by $\Theta$."* So $\dot\theta$ is the parameter-space
  motion; $\dot u_t$ is the *function-space* motion it induces; $\Theta$ is
  the bridge between them.
- **What $\Theta$ is — kernel, matrix, or operator?** All three, depending
  on how you look at it.
  $$ \Theta_\theta(x,x') = \langle \nabla_\theta f_\theta(x),\,
     \nabla_\theta f_\theta(x')\rangle_{\mathbb R^p} $$
  is, literally, *the dot product (over the $p$ parameters) of the
  network's gradient at $x$ with its gradient at $x'$*.
  - As a **function of two points** $(x,x')$ it is a **kernel** (a
    similarity between inputs, induced by the network).
  - **Evaluated on a finite set** of training points it is a **matrix** —
    the Gram matrix of the per-point gradients (this is the object SR
    actually forms; see below).
  - **Acting on the residual function** $u_t$, $\;(\Theta u)(x) = \sum_{x'}
    \Theta(x,x') u(x')$, it is an **integral operator** on function space.
    The equation $\dot u_t = -\Theta u_t$ is therefore a linear ODE *in
    function space*. So: kernel = the recipe $\Theta(x,x')$; operator =
    what it does when it acts on $u$; matrix = its restriction to your
    sample points.
- **Jacot's frozen-kernel limit (why it linearizes).** As width $m \to
  \infty$ with the right scaling: (1) $\Theta_{\theta_0}$ converges to a
  deterministic limit depending only on architecture, not on the random
  weights; (2) during training the parameters barely move ($O(1/\sqrt m)$
  each) while the function moves $O(1)$, so $\Theta_{\theta_t} \approx
  \Theta_{\theta_0}$ stays constant. Then $\dot u_t = -\Theta_0 u_t$ has the
  closed form $u_t = e^{-\Theta_0 t} u_0$.
- **What "NTK is linear" *implies*.** Diagonalize $\Theta_0 = \sum_k
  \lambda_k v_k v_k^\top$. Then $u_t = \sum_k e^{-\lambda_k t}(v_k^\top
  u_0)\,v_k$: each eigen-direction decays *independently and exponentially*
  at its own rate $\lambda_k$. Consequences: (i) the dynamics are *convex*
  (it is just kernel regression — no bad local minima for the linearized
  model); (ii) the slowest direction sets the wall-clock, rate $\propto
  \lambda_{\min}$; (iii) **spectral bias** — big-$\lambda$ (smooth,
  low-frequency) modes are learned fast, small-$\lambda$ (high-frequency,
  sharp) modes slow. This last point is the seed of Paper B's conditioning
  analysis (Slide 12).

### NTK $\leftrightarrow$ SR — what "the NTK of $\log|\Psi|$" actually means

This is the connection you said you don't get. Here it is from the ground
up.

- **One Jacobian, two ways to multiply it.** Collect the network's
  per-configuration parameter-gradients into a Jacobian
  $J_{R,i} = \partial_{\theta_i}\log|\Psi_\theta|(R)$ — rows indexed by
  configuration $R$, columns by parameter $i$. From the *same* $J$ you can
  form two different square matrices:
  - $\Theta = J J^\top$ — indexed by **configurations** $(R,R')$. This is
    the NTK / kernel: similarity *between points*.
  - $S = J^\top J$ (weighted by the sampling measure) — indexed by
    **parameters** $(i,j)$. This is the SR matrix.
  They are the two contractions of one object and share the same nonzero
  eigenvalues. "The NTK in parameter space" *is* $S$.
- **So "SR uses the NTK of $\log|\Psi|$" unpacks to:** take the function
  $\log|\Psi_\theta|$; at each MCMC-sampled configuration compute its
  gradient w.r.t. the parameters (the *score* $O_i = \partial_{\theta_i}
  \log|\Psi|$); average the outer product of these scores over $|\Psi|^2$:
  $$ S_{ij} = \mathrm{Cov}_{|\Psi|^2}[O_i, O_j]
            = \mathbb E_{R\sim|\Psi|^2}\!\big[O_i(R)\,O_j(R)\big]
              - \mathbb E[O_i]\mathbb E[O_j]. $$
  That average-of-outer-products *is* the Gram matrix of the network's
  sensitivities — i.e. the NTK, written in parameter coordinates, in the
  measure $|\Psi|^2$. It is also exactly the **quantum Fisher information
  matrix**. Three names, one object: NTK (in parameter space) = Fisher = SR
  matrix $S$.
- **What SR then does.** The plain energy gradient is $g_i = \mathrm{Cov}
  [E_L, O_i]$. SR solves $S\,\delta\theta = -g$ instead of stepping along
  $-g$ directly. Geometrically this takes steps of fixed size *in the
  function* (fixed Fisher-distance), not fixed size *in the parameters*
  (Euclidean) — it is the **natural gradient**. Because $S$ is the NTK, SR
  is rescaling exactly the slow eigen-directions that spectral bias would
  otherwise leave untrained. That is *why* SR beats plain Adam when the
  Fisher is well-estimated.

### What is $P$ in Paper B?

$P$ is the **preconditioner** — and it is the same idea as $S^{-1}$ in SR,
seen from Paper B's side. In Paper B's training analysis the number of GD
steps scales with the condition number $\kappa(A)$ of the loss Hessian $A$
(Slide 12). A preconditioner is a matrix $P$ you insert — change variables
$\hat\theta = P^{-1}\theta$, or equivalently step along $P P^\top(-\nabla
L)$ — chosen so the transformed problem $P^\top A P$ has condition number
$\approx 1$. The *ideal* $P$ satisfies
$$ P P^\top \approx A^{-1} = (L^\ast L)^{-1}, $$
i.e. $P$ is an approximate inverse-square-root of the operator stiffness
(an approximate Green's function). **SR's $S^{-1/2}$ is exactly such a
$P$.** Adam's diagonal rescaling and K-FAC's block-Fisher are cheaper,
weaker choices of $P$. So whenever Paper B writes "$P$", read "the thing we
multiply the gradient by to make the problem well-conditioned"; SR is our
concrete $P$.

**Connection.** Classical generalization (train/test split) is not the
right frame for us. We don't have a test set; we have an MCMC chain drawn
from $|\Psi_\theta|^2$. The right generalization measure is the *variance
of the local energy* $\mathrm{Var}[E_L]$. By the zero-variance principle,
$\mathrm{Var}[E_L] = 0 \iff \Psi$ is an exact eigenstate, so the
sample-fluctuation in the loss *is* the model error. There is no
decomposition because the sampling measure *is* the model. Our practical
analogue of "good generalization" is *sample efficiency*: we reach DMC
accuracy with $\sim 3 \times 10^3$ samples per SR step over $\sim 10^3$
steps — orders of magnitude fewer than typical neural VMC.

**Don't forget.** The compactness of our network ($\sim 50$ k parameters
for the correlator) is a *deliberate departure* from the wide-NTK regime.
We trade some theoretical control for the feature-learning regime, where
the manifold can move during training. The reorganization of the
correlator's leading axis across $\omega$ (slide on intrinsic dim) is only
possible because we are *not* frozen in NTK territory.

**Connection.** Classical generalization (train/test split) is not the
right frame for us. We don't have a test set; we have an MCMC chain drawn
from $|\Psi_\theta|^2$. The right generalization measure is the *variance
of the local energy* $\mathrm{Var}[E_L]$. By the zero-variance principle,
$\mathrm{Var}[E_L] = 0 \iff \Psi$ is an exact eigenstate, so the
sample-fluctuation in the loss *is* the model error. There is no
decomposition because the sampling measure *is* the model. Our practical
analogue of "good generalization" is *sample efficiency*: we reach DMC
accuracy with $\sim 3 \times 10^3$ samples per SR step over $\sim 10^3$
steps — orders of magnitude fewer than typical neural VMC.

**Don't forget.** The compactness of our network ($\sim 50$ k parameters
for the correlator) is a *deliberate departure* from the wide-NTK regime.
We trade some theoretical control for the feature-learning regime, where
the manifold can move during training. The reorganization of the
correlator's leading axis across $\omega$ (slide on intrinsic dim) is only
possible because we are *not* frozen in NTK territory.

---

## Slide 8 — A–Q2: Expressivity (depth and width)

**On the slide.** Universal approximation, the article's $\|x - y\|$
example (1 layer = exp in $d$, multi-layer = poly), the "what width is
for" remark.

### Depth vs width — the article's example, made precise

- **Universal approximation** (Cybenko 1989, Hornik 1991, Pinkus 1999).
  For any non-polynomial $\sigma$ that is locally bounded and a.e.
  continuous, $\{x \mapsto \sum_i a_i \sigma(w_i^\top x + b_i)\}$ is dense
  in $C(K)$ for compact $K \subset \mathbb R^d$. A *topological* statement.
  Hornik extended it to denseness in Sobolev norms.
- **It guarantees no rate.** Density is consistent with needing $2^{2^d}$
  neurons to achieve any positive accuracy. So universal approximation is
  necessary but extraordinarily weak.
- **The article's $\|x - y\|$ example.** Approximate the Euclidean norm
  $g(x, y) = \|x - y\|$ on $[0,1]^d \times [0,1]^d$ to error $\varepsilon$:
  - *One hidden layer with* ReLU *(or sigmoid, etc.)*: a packing argument
    shows that to distinguish all pairs $(x, y)$ at scale $\varepsilon$ in
    $\mathbb R^d$, you need to represent $\sim (1/\varepsilon)^d$ distinct
    affine functions. Width is $\Omega(2^{cd})$ for some $c > 0$ — *exponential
    in $d$*.
    - **What is the $c$?** Just a fixed positive constant that does *not*
      depend on $d$ (it comes from the packing geometry — roughly, how many
      $\varepsilon$-separated directions fit per dimension). $\Omega(\cdot)$
      is the asymptotic *lower bound* ("grows at least this fast"). The only
      thing that matters is that the exponent is *linear in $d$*: $2^{cd}$
      doubles every $1/c$ dimensions, so the required width explodes with
      dimension. The precise value of $c$ is irrelevant to the argument;
      "$2^{cd}$" just means "exponential in $d$, with some rate."
  - *Deep network*: $\|x - y\|^2 = \sum_i (x_i - y_i)^2$, computable by
    feeding the $d$ scalar differences through $d$ shared squaring
    sub-circuits (each is approximated by a small constant-width depth-$L$
    ReLU net at error $2^{-L}$ — Yarotsky 2017) and summing. Total width is
    $\mathrm{poly}(d)$, total depth is $O(\log(1/\varepsilon))$.
  - *Punchline.* The same function: exponential at depth 1, polynomial at
    depth $O(\log d)$. Depth gives *categorically* larger function classes
    at fixed budget.
- **Depth separation theorems** (Telgarsky 2016; Eldan–Shamir 2016;
  Yarotsky 2017). The sawtooth $f_L$ is representable by a depth-$L$ ReLU
  net of width $O(1)$ but requires $\Omega(2^L)$ neurons at depth $O(1)$ —
  iteration of a triangle wave. Eldan–Shamir: a radial function in
  $\mathbb R^d$ representable at depth 3 with poly-size but requiring
  $\Omega(2^d)$ at depth 2.
- **What width is for (per the article).** Two roles.
  - *Necessary* width: for ReLU, *width* $\ge d_{\rm in} + 1$ is required
    for universality (Hanin–Sellke). Below that, the network can only
    realize functions with degenerate input dependence.
  - *Manifold cross-section*: width controls the dimension of the
    feature manifold the network can vary smoothly along. Depth controls
    the *order of composition*; width controls the *richness at each
    layer*.
- **Compositionality** (Mhaskar–Poggio 2016, §1.4.1). If $f = f_K \circ
  \cdots \circ f_1$ with each $f_i$ low-dimensional ($d' \ll d$), the
  approximation rate scales with $d'$, not $d$. Depth is the formal
  mechanism by which networks beat the curse for compositional functions.

**Connection.** Our ansatz is profoundly compositional: Slater
$\circ \exp(\text{cusp} + \text{correlator}) \circ (\phi, \psi, \rho) \circ
\text{soft-core feature map}$. Each layer is small, low-dimensional, and
respects a symmetry the next layer composes with. Depth here is not "more
parameters" — it is "more composition".

**Don't forget.** Depth also gives the ResNet/ODE / message-passing
machinery — slide on backflow uses this directly.

---

## Slide 9 — A–Q3: Curse of dimensionality (Barron + intrinsic d)

**On the slide.** Curse rate $\varepsilon^{-d/s}$ with $s$ defined as
smoothness; Barron's theorem precisely; connection to MBSE; the
HF/FCI/CCSD chain; cusp/Coulomb caveat.

### What $s$ means

- $s$ is the *smoothness order* of the target function. Formally, $g \in
  C^s(\Omega)$ or $g \in W^{s,p}(\Omega)$ — the Sobolev ball of functions
  with $s$ weak derivatives in $L^p$. The classical $n$-width result says
  any method using $n$ parameters to approximate a generic element of the
  unit ball of $W^{s, p}(\Omega)$ on $\Omega \subset \mathbb R^d$ achieves
  error at best $n^{-s/d}$ — to invert, you need $n \gtrsim
  \varepsilon^{-d/s}$.
- The interpretation: *the more derivatives the target has, the faster
  classical methods converge*, but the rate is still exponential in $d$
  (only the base is improved). So pure smoothness *cannot* beat the curse.
  You need extra structure.

### Why the curse is real

The $n$-width result is saturated by *linear* methods — Fourier, polynomial,
finite-element. So for generic smooth functions in $\mathbb R^d$, no method
beats the curse. Escape routes must exploit extra structure: Barron
regularity, manifold structure, PDE structure, compositionality, symmetry.

### Barron's theorem — the proof in full so you can re-derive it on the board

**Setup.** Let $\varrho$ be ReLU or sigmoidal. For $g \in L^1(\mathbb R^d)$
with Fourier transform $\hat g$, define
$$ C_g := \int_{\mathbb R^d} \|\xi\|_2 \, |\hat g(\xi)| \, d\xi. $$
If $C_g < \infty$, then for any $n$ and any probability measure $\mu$ on
$B_1(0)$, there exists a two-layer network
$\Phi_{(d, n, 1), \varrho}(\cdot, \theta)$ of width $n$ with
$$ \inf_{\theta}\;\|\Phi(\cdot,\theta) - g\|_{L^2(\mu)} \le \frac{c \, C_g}{\sqrt n}. $$

### Every symbol in that statement (you asked about all of these)

- **$g$** — the **target function**: the thing you are trying to
  approximate, the "ground-truth answer" the network should reproduce.
  Yes — $g$ is the GT. (For us, conceptually, the smooth part of the
  correlator / wavefunction after the cusp is removed.)
- **$\hat g(\xi)$** — its Fourier transform; $\xi$ is the frequency vector.
- **$C_g$ — the Barron norm**: a single number measuring how much
  *high-frequency content* $g$ has, weighted by frequency magnitude
  $\|\xi\|$. Small $C_g$ = the function is "frequency-light" / smooth in the
  Barron sense.
- **$\inf_\theta$** — "$\inf$" is the **infimum**, the smallest achievable
  value. $\inf_\theta\|\Phi(\cdot,\theta)-g\|$ = *the best approximation
  error over all weight settings $\theta$*. The theorem is an **existence**
  statement: there *exists* a good network (it does **not** say gradient
  descent finds it — that gap is what Q4 / training is about).
- **$\Phi(\cdot,\theta)$** — the two-layer network with $n$ neurons and
  weights $\theta$; the "$\cdot$" is the input slot.
- **$\|h\|_{L^2(\mu)}$ and $\mu$** — the **root-mean-square size** of a
  function $h$, measured under the probability distribution $\mu$:
  $\|h\|_{L^2(\mu)} = \big(\int |h(x)|^2\,d\mu(x)\big)^{1/2}$. So
  $\|\Phi - g\|_{L^2(\mu)}$ is the RMS error between network and target,
  averaged over where inputs actually live. **$\mu$** is exactly that
  input distribution — "the places you care about being accurate." ($L^2$
  = mean-square norm; $\mu$ = the measure you average against.)
- **$c$** — a **universal positive constant** (does not depend on $g$, $d$,
  or $n$ — it is $\approx 2\pi$, coming out of the Monte-Carlo variance
  bound in step 3 below). Numerically unimportant; it is there so the
  inequality is honest.
- **The rate $1/\sqrt n$ is dimension-independent.** The exponent has no
  $d$ in it — error falls like $1/\sqrt n$ whether $d=2$ or $d=40$.

### What "the dimension is hidden in $C_g$" means

The curse hasn't been *deleted* — it has been *moved*. A generic-smoothness
bound puts $d$ in the **exponent** ($\varepsilon^{-d/s}$ — catastrophic).
Barron's bound has *no* $d$ in the exponent; instead $d$ can only enter
through the constant $C_g = \int_{\mathbb R^d}\|\xi\|\,|\hat g|\,d\xi$, which
is an integral over $d$-dimensional frequency space. For a *generic*
function $C_g$ still grows with $d$ — the curse is alive, just relocated.
The escape is conditional: **if** your target is genuinely Barron-regular
(its $C_g$ stays moderate even in high $d$), **then** the $1/\sqrt n$ rate
is the whole story and you have beaten the curse. The entire game is
arranging for $C_g$ to be small — which for us is what the cusp/soft-feature
conditioning buys.

**Proof sketch (article's structure).**

1. *Write $g$ via inverse Fourier.* For $x \in B_1$,
   $$ g(x) - g(0) = \int_{\mathbb R^d} (e^{i\xi^\top x} - 1) \hat g(\xi)
   \, d\xi. $$
   Take real / imaginary parts and split; the key term has the form
   $\int (\cos(\xi^\top x + b(\xi)) - \cos(b(\xi))) |\hat g(\xi)| d\xi$ for
   a phase $b(\xi) = \arg \hat g(\xi)$.
2. *Re-express as an expectation.* Define a probability measure $\mu_g$ on
   parameter space proportional to $\|\xi\|_2 |\hat g(\xi)|$ (the
   normalization is $C_g$). Then
   $$ g(x) - g(0) = C_g \, \mathbb E_{(\xi, \tilde\xi) \sim \mu_g}
   \bigl[\Gamma(\xi, \tilde\xi)(x)\bigr] $$
   for a function $\Gamma$ of fixed form (a sinusoidal-difference
   indicator).
3. *Monte Carlo over $n$ parameters.* Sample $(\xi^{(i)}, \tilde\xi^{(i)})
   \sim \mu_g$ iid for $i = 1, \dots, n$. The empirical average
   $C_g \frac1n \sum_i \Gamma(\xi^{(i)}, \tilde\xi^{(i)})(x)$
   approximates $g(x) - g(0)$. By Bienaymé's identity,
   $$ \mathbb E\|(\text{empirical mean}) - (\text{true})\|_{L^2(\mu)}^2
   = \frac{\mathrm{Var}}{n} \le \frac{(2\pi C_g)^2}{n}, $$
   so there exists at least one realization with $L^2(\mu)$ error
   $\le 2\pi C_g / \sqrt n$.
4. *Each $\Gamma(\xi, \tilde\xi)$ is well-approximated by a single neuron*
   (a sigmoidal step function — this is the only place the activation
   choice enters; it works for ReLU after a sign change).
5. Combine: $g$ is within $c C_g / \sqrt n$ of an $n$-neuron two-layer
   network in $L^2(\mu)$.

**The most beautiful single line of the article.** *The neurons are Monte
Carlo samples in frequency space.* This is the deepest connection between
deep learning and statistical physics — it lands well with quantum-physics
examiners, slow down here if they engage.

### Why Barron does not directly apply to us, and what we do about it

- The Coulomb factor $1/r$ in $E_L$ and the linear-node cusp at coalescence
  are *not* in any Barron space — their Fourier integrals diverge. So the
  pure wavefunction $\Psi$ is *not* Barron, and the bare theorem does not
  give us a no-curse rate.
- The *conditioning* trick is to take the non-Barron features out of the
  network's job:
  - The cusp factor handles $1/r$ analytically; the residue the network
    has to represent is smooth in $r$.
  - The soft-core radial features $s_k$ satisfy $ds_k/d\tilde r \to 0$ as
    $\tilde r \to 0$, so the network cannot inject a $1/r$ singularity
    through derivatives of the pair branch.
- After this conditioning, the *learned* part $W_\theta$ is — empirically
  and provably under mild assumptions — in spectral Barron space for the
  static Schrödinger equation (Chen, Lu, Lu, Zhou 2023, *SIAM J. Math.
  Anal.*, cited in Paper B's reference list). So the no-curse rate
  applies to the learned object, not the original physical object. *This
  is exactly the slide's "condition the problem to become Barron-like"
  point.*

### Intrinsic dimensionality — what we actually mean

- The configuration space is $\mathbb R^{2N}$ — $d = 40$ for $N = 20$.
- The physical solution lives on a much lower-dimensional manifold
  determined by symmetries (permutation, rotation), continuity, locality,
  and the analytic structure of the wavefunction at coalescence.
- Generalized Barron and compositional Barron spaces (E, Wojtowytsch
  2019/2020; Barron–Klusowski 2018) formalize this — they are closed under
  composition, scale to arbitrary depth, and capture the natural function
  class for neural representations of high-dimensional functions.
- *Our empirical evidence (intrinsic-dim slide)*: $r_{\rm eff}(Z) \le 3.1$
  across the entire $(N, \omega)$ grid; sample efficiency four orders of
  magnitude better than basis-expansion methods.

**Connection.** Our claim "a compact NN represents the wavefunction
efficiently" is not hope — it is a corollary of spectral-Barron regularity
theory for static Schrödinger combined with our cusp-engineering. We do
not compute $C_g$ for our $\Psi$ (open problem), but the existence theorem
gives us a no-curse rate, and the empirical manifold has $r_{\rm eff} \le
3$.

**Don't forget.** Say the words *"dimension-independent"* and *"Monte Carlo
over neurons"*. This is the moment to be slow.

---

## Slide 10 — A–Q4: Optimization

**On the slide.** *(The NTK linearization was moved to Slide 7 — do not
re-derive it here.)* This slide carries four things only: the non-convex
landscape and **saddle escape**, **implicit regularization**, **Adam**, and
**natural gradient** (= SR for us); plus architecture-as-optimization-aid.

### The four things on the slide, in one breath each

- **Non-convex landscape + saddle escape.** The loss has exponentially many
  critical points; worst-case theory says SGD cannot find a global minimum.
  In practice it does, because the critical points are dominated by
  *saddles*, not bad local minima (Dauphin 2014), and the *stochastic
  noise* in SGD kicks the iterate off saddles. So non-convexity is not the
  obstacle classical theory feared.
- **Implicit regularization.** Among the infinitely many parameter settings
  that interpolate the data, (S)GD started from small initialization drifts
  to the *minimum-norm / flat* one. That is a free bias toward simple
  solutions — it is *why* huge networks generalize without an explicit
  penalty. (This is the "implicit bias" idea; it belongs here on the
  optimization slide, not on Q1.)
- **Adam.** Per-coordinate adaptive preconditioner — see below.
- **Natural gradient = SR.** The Fisher-metric update — see below; for us it
  is Stochastic Reconfiguration, the link you set up on Slide 7.

### Optimizers — what each one does, and why each one matters

- **SGD (stochastic gradient descent).** $\theta_{t+1} = \theta_t - \eta
  \widehat{\nabla L}(\theta_t)$, where the gradient is estimated on a
  *batch* of size $B \ll n$. The estimator is unbiased ($\mathbb E[\hat
  \nabla L] = \nabla L$) but noisy. Two effects of the noise:
  - *Implicit regularization*: SGD noise is approximately isotropic in
    flat regions of the loss landscape but anisotropic in stiff
    directions; this biases the solution toward flat minima, which
    generalize better (Keskar 2017, Jastrzębski 2017). For us this is the
    analogue of "implicit bias picks the min-norm interpolant" — only now
    it's a flat-minimum interpolant.
  - *Saddle escape*: deterministic GD gets stuck at saddles; noise kicks
    you off. Dauphin 2014: most critical points of deep losses are
    saddles, not local minima.
  - **How batching helps**: a too-small batch gives a gradient estimate
    too noisy to make progress; a too-large batch gives a gradient too
    accurate, which means *less* implicit regularization and worse
    generalization. The "right" batch size is the one where the noise
    scale matches the curvature scale of the loss (linear scaling rule,
    Goyal 2017).
- **Adam.** $\theta_{t+1} = \theta_t - \eta \hat m_t / (\sqrt{\hat v_t} +
  \varepsilon)$, with $\hat m_t$ the EMA of $\nabla L$ and $\hat v_t$ the
  EMA of $(\nabla L)^2$ — both per coordinate. The denominator is a
  *diagonal preconditioner*: each coordinate is scaled by the running
  variance of its gradient. If the loss landscape has axis-aligned
  curvature variation, Adam corrects for it. It does *not* correct for
  off-diagonal curvature (rotated stiffness).
  - **Adam vs SGD.** Adam handles per-coordinate scale much better — it is
    very robust to learning-rate choice and to badly scaled inputs. It
    pays back in two ways: (a) sometimes worse final generalization
    because the diagonal preconditioning weakens the implicit bias; (b)
    extra memory ($2p$ for the moments). For our PINN, Adam is the
    workhorse at small $\omega$ (where the Fisher matrix becomes
    rank-deficient and SR fails); SR is the workhorse at moderate-to-large
    $\omega$.
- **Natural gradient (one update, two names).** *Read this carefully — the
  letter $F$ on the slide and the letter $S$ in the notes are the same
  matrix.*
  - Natural gradient (Amari, ML community): $\theta_{t+1} = \theta_t - \eta
    F^{-1}\nabla L$, where $F$ is the **Fisher information matrix**.
  - Stochastic Reconfiguration (Sorella, VMC community): $\theta_{t+1} =
    \theta_t - \eta S^{-1}g$, where $S$ is the **SR matrix**.
  - In our setting **$F = S$** identically — they are the *same object*,
    just different community notation:
    $$ S_{ij} = F_{ij} = \mathrm{Cov}_{|\Psi|^2}\!\big[O_i, O_j\big],
       \qquad O_i = \partial_{\theta_i}\log|\Psi_\theta|. $$
    Equivalently, the *empirical NTK of $\log|\Psi|$ written in parameter
    coordinates under the sampling measure $|\Psi|^2$*.
  - The energy gradient is $g_i = \mathrm{Cov}_{|\Psi|^2}[E_L, O_i]$, and
    the SR step is the linear solve $S\,\delta\theta = -g$ (done by CG
    because $S$ is $p\times p$ and dense — you never form $S^{-1}$).
  - **Geometric interpretation.** Take steps of fixed *Fisher-distance*
    (small change in the function $\Psi$), not fixed Euclidean distance
    (small change in $\theta$). Parameterization-invariant: reparametrize
    the network and the natural-gradient step is the *same* step.
  - **Connection to Paper B's preconditioning.** $S^{-1/2}$ is exactly the
    matrix $P$ from Slide 12 — it counter-shapes the parameter-space
    curvature $TT^*$. SR is not just *like* natural gradient; it *is*
    natural gradient with $F = S$.
  - **Failure mode.** When ESS drops, $S$ becomes rank-deficient (the
    sampling measure doesn't span the parameter directions); CG diverges
    or the regularization $\varepsilon I$ dominates, and SR effectively
    reduces to plain gradient descent. Adam is the robust fallback because
    it only needs $\nabla L$, not its full covariance.

### (NTK lives on Slide 7 now — pointer only)

If asked "where does the NTK come in here", point back to Slide 7: training
linearizes into kernel regression, the spectrum of $\Theta$ sets
convergence, and *spectral bias* (below) is the prediction that connects
this slide to Paper B's conditioning story. Do **not** re-derive $\dot u_t =
-\Theta u_t$ on this slide.

### Spectral bias (for notes only, not on slides)

Eigendecompose $\Theta^\infty = \sum_k \lambda_k v_k v_k^\top$. The
residual decays along $v_k$ at rate $e^{-\lambda_k t}$. For smooth
kernels, $\lambda_k$ tracks the eigenvalue density of a Laplacian-like
operator on the input distribution; the eigenfunctions $v_k$ are the
input-space "frequencies". Low-frequency $\Rightarrow$ large $\lambda_k$
$\Rightarrow$ fast decay. High-frequency $\Rightarrow$ slow decay.

Why this matters for PINNs (Paper B angle, not on this slide but be ready):
the PDE residual involves derivatives, which act on the same eigenbasis as
multiplications by powers of $|\mathbf k|$ in Fourier space. So
high-frequency targets become *much* harder under a derivative loss than
under a supervised loss — the residual has to suppress more high-frequency
content, against the NTK's bias. *This is Paper B's slide 12 result said
backward.*

### Architecture as optimization aid

- *ResNets* (He 2016). $\bar \Phi^{(\ell)} = \varrho(\Phi^{(\ell)}) +
  \bar \Phi^{(\ell - 1)}$ is an Euler step of the ODE $\dot \phi = h(t,
  \phi)$. Cures vanishing/exploding gradients: the Jacobian is close to
  identity at init.
- *CNN / GNN equivariance*. Restricting to functions equivariant under a
  symmetry group shrinks the hypothesis class — the NTK acts on a smaller
  space, the spectrum is more favorable, and the worst eigenvalues are
  gone.

**Connection.** Our optimizer *is* SR (= NTK natural gradient in QMC dress).
Our backflow *is* a near-identity ODE flow (ResNet). Our correlator *is* a
permutation-equivariant Deep Sets / GNN. Every Paper-A optimization trick
is named, and we use it.

**Don't forget.** SR = QMC NTK = natural gradient. Be able to walk anyone
through: $\partial_\theta \log|\Psi|$ is the score; its covariance is
Fisher; Fisher is the natural-gradient preconditioner; and the empirical
Fisher *is* the NTK in the sampling measure.

---

## Slide 11 — Paper B Q1–Q3 (one slide, compact)

**On the slide.** The error decomposition, three bullets for Q1/Q2/Q3, the
red box on the headline ($L^*L$ binding bottleneck).

### Q1 — approximation error (brief)

- The question: can a $\tanh$-network of width $W$ and depth $L$ contain a
  function with small PDE residual?
- Answer (De Ryck–Mishra–Jagtap): yes, with rate $\varepsilon^{-d/(k-j)}$
  for $u \in H^k$, $j < k$ derivatives needed. Why $\tanh$: PDE residuals
  need $j \ge 2$ derivatives, ReLU isn't $C^2$.
- **What a "Sobolev rate" is (you asked).** A *Sobolev norm* $\|\cdot\|_{H^j}$
  measures a function *together with its first $j$ derivatives* — not just
  its values. The rate $\varepsilon^{-d/(k-j)}$ then reads: *to drive the
  error — measured in $H^j$, i.e. including $j$ derivatives — below
  $\varepsilon$, the network width must grow like $\varepsilon^{-d/(k-j)}$,
  given the target has $k$ derivatives ($u\in H^k$).* The reason PINNs need
  a Sobolev (not plain $L^2$) rate: the PDE residual *applies derivatives*
  to the network, so being close in values isn't enough — you must be close
  in derivatives too.
  - **The constants** ($C$, $W$, $L$ hidden in the bound) are generic
    positives depending on the domain, the smoothness $k$, and the
    dimension $d$ — *not* on $\varepsilon$. Standard big-$O$ bookkeeping;
    don't memorize them.
- This is *just* universal approximation in a Sobolev norm. Paper A
  already covers it; Paper B states it cleanly and uses it.
- The curse is *still* there ($d$ in the exponent). Beating it needs extra
  structure: Barron, manifold, PDE — same escape routes as Paper A.
- **For us**: our network class has antisymmetry (Slater determinant),
  permutation invariance (Deep Sets), and an analytic cusp. We are in a
  strict subspace of the bare $\tanh$-network class — approximation is
  *easier*, not harder, than the bare theorem says. Empirically: $\sim
  50$k parameters in the correlator suffice across the entire $(N, \omega)$
  grid.

### Q2 — stability (brief)

- Stability inequality: $\|u_\theta - u\|_{\mathcal X} \le C_{\rm stab}
  \|\mathcal L u_\theta - f\|_{\mathcal Y}$, with $C_{\rm stab} =
  \|\mathcal L^{-1}\|$.
- For us: $\mathcal L = \hat H - E_0$ is not invertible at the eigenvalue
  (kernel = $\Psi_0$). $C_{\rm stab}$ formally infinite on the eigenspace,
  $1/\text{gap}$ on its complement.
- Resolution: change the form — Rayleigh quotient ($\langle \Psi | \hat H
  | \Psi \rangle / \langle \Psi | \Psi \rangle$, the *Deep Ritz* method).
  Stability automatic via the variational principle. Paper B's recurring
  lesson: *"regularity dictates the form."*
- For our problem this is trivial in disguise: zero-variance principle says
  $\mathrm{Var}[E_L] = 0 \iff \Psi$ is an exact eigenstate. So a small
  generalization gap implies a small training loss, and a small training
  loss with bounded variance implies a perfect eigenstate. Both losses
  collapse together at the ground state.

### Q3 — generalization / quadrature (brief)

- The PINN training loss is a Monte Carlo estimate of an integral; the
  question is how close the MC sample mean is to the integral.
- Standard Rademacher / Dudley concentration:
  $$ \sup_{\theta} |L_{\rm train} - L_{\rm true}| \lesssim \mathrm{Lip}
  (\Theta) \sqrt{\log \mathcal N(\Theta) / n}. $$
  - **$\sup_\theta$ — the supremum** ("the largest value over all
    $\theta$"). $\sup_\theta|L_{\rm train}-L_{\rm true}|$ is the *worst-case*
    gap between the finite-sample loss and the true loss, over *every*
    parameter setting the network could land on. Bounding the sup means the
    guarantee holds *no matter where* training ends up — you don't get to
    assume it landed somewhere nice.
  - **$\mathrm{Lip}(\Theta)$ — the Lipschitz constant** of the loss as a
    function of the parameters. A function is Lipschitz-$L$ if
    $|h(a)-h(b)| \le L\,|a-b|$: it can't change faster than $L$ per unit
    step. Here it bounds how *wiggly* the loss is across parameter space —
    a wigglier loss needs more samples to estimate uniformly, which is why
    it multiplies the bound. ($\mathcal N(\Theta)$ is a covering number — how
    many small balls tile the reachable parameter set; $n$ is the number of
    collocation samples.)
### The assumptions behind that bound — and whether they hold for our SD ansatz (you asked)

This is the right thing to interrogate, because the bound is only as good
as its hypotheses. Paper B's quadrature/generalization bound needs **three**
things:

1. **Bounded integrand.** The quantity being averaged (the residual, or for
   us the local energy $E_L$) must be bounded, or concentration constants
   blow up.
2. **Lipschitz-in-$\theta$ loss with a controllable constant.** The map
   $\theta \mapsto L(\theta)$ must not change arbitrarily fast as you move
   the weights — that is the $\mathrm{Lip}(\Theta)$ factor.
3. **Compact / bounded parameter set $\Theta$**, so it has a finite
   covering number $\mathcal N(\Theta)$.

- **Are neural networks assumed Lipschitz? Under what conditions?** Yes —
  Lipschitzness is an *assumption on the parametrization*, and it holds for
  the networks Paper B analyses: $\tanh$ activations (and their derivatives)
  are bounded, and a composition of finitely many bounded-derivative layers,
  **restricted to a bounded weight domain $\Theta$**, is Lipschitz in
  $\theta$. The catch specific to PINNs: the loss contains *derivatives*
  $\mathcal L q$, and each derivative multiplies the Lipschitz constant by
  roughly a frequency factor — so $\mathrm{Lip}(\Theta)$ grows with the
  derivative order of the PDE and with the weight norms. (This is the same
  high-frequency penalty that reappears as $L^*L$ stiffness on Slide 12.)
- **Do these assumptions survive when we use a Slater-determinant ansatz on
  the MBSE?** Partly — and the place they *fail* is exactly what our
  engineering fixes:
  - *Lipschitz-in-$\theta$ — holds:* the determinant is a smooth (polynomial)
    function of the orbital values, and the orbitals are smooth in their
    parameters, so on a bounded parameter domain $\theta \mapsto \Psi_\theta$
    is smooth and Lipschitz. The SD does **not** break assumption 2.
  - *Bounded integrand — fails for the bare ansatz:* this is the one that
    breaks. The local energy $E_L = H\Psi/\Psi$ blows up at electron
    coalescence (the Coulomb $1/r$), and naively one worries about the
    nodes where $\Psi = 0$. So the bare SD+Coulomb ansatz violates
    assumption 1 and Paper B's bound does **not** apply to it directly.
  - *Our fix:* the analytic cusp cancels the $1/r$ in $E_L$ exactly, so
    $E_L$ stays finite at coalescence; near the nodes $E_L$ is actually
    finite for a smooth fixed-node wavefunction (numerator and denominator
    vanish together, the node is integrable), and the soft-core features
    stop the network from re-introducing a singularity through derivatives.
    *With* the cusp + soft features, assumption 1 is restored and the bound
    applies. So the SD is fine; the binding issue was boundedness of $E_L$,
    and that is precisely what we engineer.
- **What is the covering number $\mathcal N(\Theta)$?** It is a *complexity
  measure of the parameter set*: the number of small balls of radius
  $\delta$ needed to cover $\Theta$. It enters because the $\sup_\theta$ is
  controlled by a union bound over a finite $\delta$-net of $\Theta$ —
  $\log \mathcal N$ is, loosely, "effective number of parameters $\times
  \log(1/\delta)$." A bigger / more flexible network $\Rightarrow$ larger
  $\mathcal N$ $\Rightarrow$ you need more samples $n$ to make the train-vs-true
  gap small. It is *not* anything quantum or VMC-specific; it is the
  classical statistical-learning capacity term.
- **Is this uncertainty "just for VMC"?** No. The gap $|L_{\rm train} -
  L_{\rm true}|$ is generic to *any* loss estimated by sampling — it applies
  equally to the **collocation/PINN** pipeline (importance-sampled
  collocation points) and to **VMC** (MCMC samples from $|\Psi|^2$). What is
  VMC-specific in our work is the *remedy*, not the problem: we make the
  reported numbers immune to this gap by certifying the final energy with an
  **independent heavy-VMC evaluation** ($\sim$30 k samples), so the
  training-time sampling bias never leaks into a quoted result. So: the
  quadrature uncertainty lives in *both* training pipelines; the heavy-VMC
  pass is how we measure the truth at the end.
- **Practical metric: effective sample size (ESS).** When ESS collapses at
  small $\omega$ or large $N$, the effective $1/\sqrt n$ becomes $1/\sqrt{\rm
  ESS}$, which is the binding constraint on collocation training.

**Don't forget.** Quote the headline almost verbatim: *"the binding
bottleneck in physics-informed machine learning is training, governed by
the spectrum of the Hermitian square of the differential operator."* This
is the central sentence of Paper B and your bridge to slide 12.

---

## Slide 12 — B–Q4: Training = conditioning of $L^* L \circ TT^*$

**On the slide.** Linearized GD; $\kappa(A)=\lambda_{\max}/\lambda_{\min}$
controls training cost; Theorem 7.10 ($\kappa$ factorizes as $L^*L \circ
TT^*$); Laplacian example $\kappa \sim (k_{\max}/k_{\min})^4$; the
preconditioning principle. *(The explicit step-count $\#(\varepsilon) =
O(\kappa\ln 1/\varepsilon)$ was dropped from the slide — keep it in your head
as the reason $\kappa$ matters; it is explained below for your own use.)*

### Setup (Paper B §7.2)

Linearize around initialization $\theta_0$. Define $q_i(x) =
\partial_{\theta_i} u_{\theta_0}(x)$ — the per-parameter feature map. To
second order in $\theta - \theta_0$:
$$ u_\theta(x) \approx u_{\theta_0}(x) + q(x)^\top (\theta - \theta_0). $$
The loss becomes a quadratic in $\theta$ with Hessian
$$ A_{ij} = \langle \mathcal L q_i, \mathcal L q_j \rangle_{L^2(\Omega)}
+ \lambda \langle q_i, q_j\rangle_{L^2(\partial \Omega)}. $$
This is the *Hessian of the linearized PDE-residual loss in parameter
space*.

### What $\kappa, \lambda, \mathbf A, TT^*$ are

- $\lambda_{\max}, \lambda_{\min}$ are the largest and smallest eigenvalues
  of $A$ (or equivalently, of the operator $A$ acts as).
- $\kappa(A) = \lambda_{\max} / \lambda_{\min}$ is the *condition number*.
- $\mathbf A = L^* L$ is the *Hermitian square* of the differential
  operator $L$. For self-adjoint $L$ (Schrödinger, Laplacian), $L^* L =
  L^2$.
- $T : \mathbb R^p \to L^2(\Omega)$, $T \delta\theta = q^\top \delta\theta$
  — *lifts* a parameter perturbation into function space.
- $T T^* : L^2(\Omega) \to L^2(\Omega)$ is the *kernel integral operator
  of the neural tangent kernel*: $(T T^* f)(x) = \int q(x)^\top q(x')
  f(x') \, dx'$, with kernel $\Theta(x, x') = q(x)^\top q(x')$.
- So $T T^*$ is *the NTK as an operator on function space*. (And $T^* T$
  is the corresponding $p \times p$ matrix on parameters, which is exactly
  what Adam / K-FAC / SR estimate.)
- **What the "$\circ$" means in $L^*L \circ TT^*$.** It is *operator
  composition* — "apply one operator, then the other" — not number
  multiplication and not an element-wise (Hadamard) product. Both $L^*L$ and
  $TT^*$ are linear operators on the function space $L^2(\Omega)$, so
  $L^*L \circ TT^*$ is the operator $f \mapsto L^*L\big(TT^* f\big)$. (If you
  represent the operators as matrices, composition is just matrix
  multiplication; the circle is the coordinate-free way to write it.) The
  theorem's content is that the condition number GD feels equals the
  condition number of this *composed* operator — the PDE stiffness $L^*L$
  acting through the network curvature $TT^*$.

### Theorem 7.9 — gradient descent on a quadratic, and what $\#(\varepsilon)$ means

**Reading the formula first.** $\#(\varepsilon)$ is *the number of
gradient-descent steps* needed to reach error $\varepsilon$. The theorem
says
$$ \#(\varepsilon) = O\big(\kappa(A)\,\ln(1/\varepsilon)\big), $$
which reads: *steps-to-converge $=$ (condition number) $\times$ (how many
$e$-folds of accuracy you want).* Two separate dependencies:
- the $\ln(1/\varepsilon)$ factor is **cheap** — each factor-of-$e$
  reduction in error costs a *constant* number of extra steps (linear
  convergence);
- the $\kappa(A)$ factor is **expensive** — it *multiplies* your whole
  iteration count. Double the condition number, double the training time.
So $\kappa$ is the one quantity that controls cost at scale.

**Where it comes from.** The error contracts as $(1 - \eta A)(\theta -
\theta^*)$, so each eigendirection decays by $1 - \eta\lambda$. The stable
step is $\eta = c/\lambda_{\max}$ ($c<1$, capped by the *stiffest*
direction). The *slowest* direction then decays by $1 - c\lambda_{\min}/
\lambda_{\max} = 1 - c/\kappa$ per step — so you need $\sim \kappa$ steps
per $e$-fold. **At large scale, only the condition number matters.**

### Theorem 7.10 — the factorization

$\kappa(A) = \kappa(\mathbf A \circ T T^*)$ where $\mathbf A = L^* L$. So
the stiffness GD feels factors as:
- $L^* L$ = *operator-level* stiffness from the PDE — fixed once you
  picked the equation.
- $T T^*$ = *parameter-level* curvature from the network — depends on
  architecture, init, training trajectory.

Their *composition* is what controls $\#(\varepsilon)$.

### What *is* $L$, and how is it the "stiffness of the PDE"?

- **$L$ is the differential operator of the PDE** — the thing on the
  left-hand side of $Lu = f$ that acts on your unknown function. For
  Poisson, $L = -\Delta$. For us the operator inside the energy residual is
  built from the kinetic Laplacian $-\tfrac12\Delta$. "$L$" is *the
  equation, written as an operator.*
- **"Stiffness" = how violently $L$ amplifies high-frequency content.**
  Gradient descent doesn't see $L$ directly; it sees $\mathbf A = L^*L$
  (the loss is a *squared* residual, so the Hessian carries $L$ twice). The
  spread of eigenvalues of $L^*L$ *is* the stiffness: if the operator
  multiplies fine wiggles by a huge factor and smooth parts by a tiny
  factor, then $\lambda_{\max}/\lambda_{\min} = \kappa$ is huge and GD
  crawls. "Stiff PDE" $\equiv$ "$L^*L$ has a wide eigenvalue spread"
  $\equiv$ "ill-conditioned training."

### Why we go to Fourier (and why Laplacian gives $\kappa \sim k^4$)

**Why Fourier at all:** the Laplacian is *diagonal* in the Fourier basis —
plane waves are its eigenfunctions, $-\Delta\,e^{i\mathbf k\cdot x} =
|\mathbf k|^2 e^{i\mathbf k\cdot x}$. So in Fourier space the operator stops
being a differential operator and becomes plain *multiplication by a
number*, $|\mathbf k|^2$. That makes its eigenvalues readable at a glance —
which is exactly what you need to compute a condition number. Fourier turns
"spectrum of a differential operator" into "range of a multiplier."

Then $L = -\Delta \Rightarrow L^*L = \Delta^2$ (the bi-Laplacian), whose
Fourier multiplier is $|\mathbf k|^2$ squared $= |\mathbf k|^4$. If the
content the network must represent spans frequencies $[k_{\min}, k_{\max}]$,
$$ \kappa \sim \Big(\frac{k_{\max}}{k_{\min}}\Big)^4. $$
**Quartic growth.** With $k_{\max} \sim 1/\ell$ for the smallest resolved
length scale $\ell$, $\kappa$ explodes for sharp nodes, cusps, or localized
structure — exactly our hard regimes.

### Link to Coulomb and our Laplacian

- Coulomb $1/r$ has a Fourier transform $\sim 1/|\mathbf k|^2$ — *all*
  frequencies present with $1/|\mathbf k|^2$ amplitude. Squaring (in $L^*
  L$) does not regulate this.
- Our kinetic term is $-\frac12 \Delta \log |\Psi| + \frac12 \|\nabla
  \log|\Psi|\|^2$. The Laplacian inside the local energy is exactly the
  $L$ whose $L^* L$ gives $k^4$. *That* is the operator-level stiffness
  for our problem.
- The cusp condition kills the worst high-frequency contribution from $V =
  1/r$ analytically. Soft-core features cap $k_{\max}$ on the learnable
  side. Together they reduce $k_{\max}/k_{\min}$ from "as bad as your
  smallest sample distance" to "as bad as your softening scale".

### Why backflow is specifically a problem here

- Backflow shifts $\tilde R \to \tilde R + \Delta_\beta(\tilde R)$ *inside
  the Slater determinant*.
- The local energy depends on $\nabla \log \Psi$ and $\Delta \log \Psi$.
  Both involve derivatives through $\Delta_\beta$. So
  $\nabla_\theta E_L$ involves *second* parameter-space derivatives of
  $\Delta_\beta$ — the gradient signal flows through the Laplacian of the
  backflow's parameter dependence.
- This couples backflow parameters directly to $L^* L$ stiffness. Adam,
  K-FAC, SR all touch only $T T^*$ (parameter-space curvature); they
  cannot reach the operator stiffness sitting in $L^* L$.

### The catch-22 result (slide 13)

For $N = 6$, $\omega = 0.5$, with the backflow ansatz on the collocation
pipeline:

- 17 different interventions (hard mining, smoothness penalty, SIR,
  Gumbel top-K, hybrid loss, det filter, soft Coulomb, kinetic clipping,
  K-FAC, Adam burst, …): all give error in $[0.32\%, 24.7\%]$.
- Same ansatz under VMC + SR: $< 0.01\%$.
- **What actually worked**: remove the Laplacian path from the backflow's
  parameter dependence — keep the kinetic term in the loss (so the loss
  still measures the right thing), but route the backflow's gradients
  through REINFORCE only (which differentiates through the score
  $\nabla_\theta \log |\Psi|$, *not* through the Laplacian). The
  operator-level coupling between backflow parameters and $L^* L$ vanishes;
  the rest of the conditioning machinery (cusp, soft features, SR) handles
  what's left.
- *This is the strongest single piece of evidence we have for Theorem
  7.10 being predictive.* Necessary-but-not-sufficient holds: parameter-
  space preconditioning is necessary (without SR or Adam nothing works at
  all), but it is not sufficient when the operator is fundamentally stiff.

### Why our analytic cusp helps so much

Two angles.

- *Approximation*: $1/r$ is not in any Barron space; trying to learn it
  needs exponentially many parameters. The cusp factor removes it from
  the network's job. *Now* the function the network actually has to
  represent is smooth.
- *Conditioning*: the cusp factor reduces $k_{\max}$ in $L^* L$ from
  "as bad as your closest sample" to "as bad as your softening scale".
  $\kappa$ drops by orders of magnitude.

### When we "condition the problem", what *exactly* are we conditioning?

We are shrinking the condition number $\kappa(A) = \kappa(L^*L \circ
TT^*)$ — the eigenvalue spread of the matrix gradient descent actually
descends on. There are exactly two handles, because $\kappa$ has two
factors:

1. **Reduce the operator stiffness $L^*L$** — i.e. shrink the *range of
   frequencies that matter*, $k_{\max}/k_{\min}$. This is what the
   *problem-side* tricks do, and they act before any optimizer is chosen:
   - *trap units* $\tilde r=\sqrt\omega\,r$ fix the length scale $\ell\to1$,
     so $k_{\max}/k_{\min}$ doesn't blow up as $\omega$ changes;
   - *soft-core features* cap how high a frequency the learned part can
     inject ($k_{\max}$);
   - *analytic cusp* removes the $1/r$ contribution entirely (the worst
     high-frequency offender);
   - *changing the form* (variational / REINFORCE) lowers the derivative
     order of the operator, so $L^*L$ is milder to begin with.
2. **Reshape the network curvature $TT^*$** to counter whatever stiffness
   remains — i.e. multiply the gradient by a $P$ with $PP^\top \approx
   (L^*L)^{-1}$. This is what the *optimizer-side* preconditioners do:
   - Adam / RMSProp: diagonal $P$ from the second moment of the gradient
     (helps if stiffness is axis-aligned);
   - K-FAC: block-diagonal Kronecker-factored Fisher;
   - SR / natural gradient: $P \approx S^{-1/2}$ with $S$ the quantum
     Fisher $=$ NTK in the sampling measure;
   - Fourier-feature embeddings (Paper B's Poisson example): pick features
     $q$ that *diagonalize* $L^*L$, with $P_{kk}=1/k^2$ exactly cancelling
     the $k^4$.

So "conditioning" is not one thing — it is *either* squeezing the frequency
range in $L^*L$ *or* counter-shaping $TT^*$. The catch-22 (Slide 13) is
precisely that the optimizer-side handle (2) cannot fix a pathology that
lives in the operator-side factor (1).

### "Supervised learning is the special case $L = \mathrm{Id}$, $\mathbf A = I$" — what it does and does **not** claim

You flagged this as sounding inaccurate. Here is the precise meaning.

- In *ordinary supervised learning* you fit $u_\theta(x) \approx y(x)$
  directly: you compare the network's *values* to targets, with **no
  derivatives applied**. The operator relating your output to what you
  score against is therefore the identity, $L = \mathrm{Id}$, so $\mathbf A
  = L^*L = I$.
- Substitute $\mathbf A = I$ into Theorem 7.10: $\kappa(A) = \kappa(I \circ
  TT^*) = \kappa(TT^*)$. **All that remains is the NTK factor.**
- So the precise claim is *not* "supervised learning is easy." It is:
  *supervised learning has **no operator-stiffness contribution** — its
  conditioning is purely the NTK $TT^*$.* Supervised training can still be
  badly conditioned if $TT^*$ itself is bad (poor architecture, bad
  scaling) — that part is shared with PINNs.
- The point of the statement is the **difference**: *everything a PINN
  suffers beyond a supervised problem is the extra factor $L^*L \ne I$* —
  the derivatives in the PDE. That is why PINNs are categorically harder to
  train than fitting data, and it is the whole reason Paper B needs a Q4 at
  all. Phrase it that way and it is exactly right.

**Connection.** Our appendix derives this *independently* in our own
notation and cites Paper B at every step. The bi-Laplacian appears
explicitly. With $k_{\max} \sim 1/\ell$ for the smallest length scale that
needs representation, $\kappa$ explodes at small $\omega$ and sharp nodes
— exactly where we observed slowest training. The fix (change the form +
cusp + SR) collapses $\kappa$ in the regimes where the catch-22 hits.

**Don't forget.** Three numbers to be ready to write on the board: $\kappa
\sim (k_{\max}/k_{\min})^4$ for the Laplacian; supervised $=$ ($L = \mathrm
{Id}$, $\mathbf A = I$, so only $TT^*$ remains); preconditioning $=$ pick
$P$ with $TT^* \approx (L^*L)^{-1}$.

---

## Slide 13 — Catch-22: $TT^*$-preconditioning is not enough when $L^*L$ is stiff

Already covered in depth in the slide-12 notes above. On the slide
itself, the talking points are: setting, 17 interventions, the same
ansatz under VMC+SR succeeds, what helped, the negative-form Theorem 7.10
statement, the cusp remark.

**On the slide language to read off if you want a clean phrasing.**
*"This is Theorem 7.10 in negative form: K-FAC and Adam precondition $TT^*$,
i.e. parameter-space curvature. The stiffness sits in $L^*L$, the
operator. Necessary but not sufficient. The only thing that worked was to
change the form — REINFORCE removes $\nabla_\theta$ through the Laplacian,
and that's exactly the coupling between backflow parameters and the
operator stiffness."*

---

## Slide 14 — SR + VMC results: PINN + CTNN matches DMC across the phase diagram

**On the slide.** The relative-error figure, the energy table, the CTNN
ablation numbers (22 / 30 / 12 % at $\omega = 1 / 0.1 / 0.001$), the
"entire increase is kinetic" remark.

**Material.**

- **What the table shows.** PINN+BF and PINN+CTNN against DMC reference:
  - $N = 2$: both within DMC error bars across all $\omega$.
  - $N = 6$: $10^{-4}$ to $10^{-3}$ relative error for both BF and CTNN;
    CTNN better. Bolded numbers in the CTNN column highlight where it
    beats BF.
  - $N = 12$: $10^{-3}$ relative error; CTNN better.
  - $N = 20$: no other neural benchmark for SR+VMC; CTNN matches DMC at
    $\omega = 0.5, 1.0$ to $\sim 4 \times 10^{-5}$ relative.
- **Compute efficiency.** $\sim 3 \times 10^3$ samples per SR step, $\sim
  10^3$ steps total. Orders of magnitude less than typical neural VMC
  ($10^5$–$10^6$ steps). The implicit-bias / generalization story from
  slide 7 in action.
- **CTNN ablation procedure.** Zero the message-passing weights of the
  CTNN cell while keeping every pairwise feature intact. Retrain from
  scratch.
- **The kinetic-only signature.** $\Delta V$ is statistically *zero* at the
  same MCMC sample positions. The entire $\Delta E$ comes from $\Delta T$.
  Physics: backflow improves the nodal surface (where $\Psi$ changes
  sign); changing the nodes changes $T = -\frac12 \langle \Delta \log
  |\Psi| \rangle$ but not $V$ (a multiplicative operator). The
  message-passing cell of the CTNN is doing what backflow is *supposed*
  to do — improving nodal geometry, not amplitudes between nodes. This
  is the cleanest single statistical fact in the thesis.
- **Wigner collapse to 11.8 %.** In the deep Wigner regime, electrons
  localize into a classical lattice; the correlation structure is
  dominated by pair distances plus the lattice constraint; three-body
  correlations exist but are not separately needed once the lattice
  geometry is fixed. The cell's expressivity is present but unused — there
  is no beyond-pairwise structure left to learn. (Three-body slide makes
  this precise.)

**Connection.** This is the SR + VMC half of "what we built actually
works". The PINN/collocation half is on the next slide.

**Don't forget.** *Kinetic only* is the cleanest statistical fact. Repeat
it. The potential energy is statistically identical at the same sample
positions, full stop. This proves we are measuring an architectural
effect on the nodal geometry, not noise.

---

## Slide 15 — PINN / collocation results: MCMC-free training of the CTNN

**On the slide.** The collocation results table, the "tricks that
mattered" bullets, the "where SR breaks down" remark.

**Material.**

- **What collocation gives you.** $E$ in Hartree from the collocation
  pipeline at $N = 6, 12, 20$. Best $N{=}6, \omega{=}1.0$: $+0.013\%$
  error against DMC. Sub-$0.25\%$ across all five $\omega$ at $N{=}6$.
  $N{=}12$: $+0.018$ and $+0.028\%$ at $\omega = 1.0, 0.5$.
  $N{=}20$: $+1.45\%$ (Jastrow-only) at $\omega = 1.0$.
- **What "MCMC-free training" buys.** No autocorrelation; no burn-in; no
  mixing-time problem at large $N$ or small $\omega$; embarrassingly
  parallel; every failure mode shows up directly in the gradient signal
  rather than hiding in sampling statistics. The cost is the ESS of the
  collocation proposal against $|\Psi|^2$.
- **Tricks that mattered.**
  - *Three-stage cascade*: joint REINFORCE pretraining (900 epochs,
    lr $5 \times 10^{-4}$) → low-LR continuation (750 epochs, lr $3
    \times 10^{-4}$) → hard-focus polish (500 epochs, lr $2 \times
    10^{-4}$). Each stage inherits the full checkpoint. Monotone
    improvement per stage: $+0.08\% \to +0.05\% \to +0.009\%$.
  - *Importance-resampling*: keep 4096 collocation points from $8\times$
    oversampled mixture proposal. The mixture has analytic Gaussian
    components tuned to the physics; the oversampling cushions ESS
    collapse.
  - *REINFORCE only*: no $\nabla_\theta$ path through the Laplacian for
    backflow gradients. This is the catch-22 fix from slide 13 — it is
    what makes collocation training of backflow even possible.
  - *Rollback gate*: per-epoch variance check; reject the update if
    variance grows above a threshold. Catches ESS-collapse failure modes
    early.
- **Where SR breaks down.** $\omega \le 0.01$. The Fisher matrix becomes
  ill-conditioned (eigenvalues span many orders of magnitude because the
  wavefunction's sensitivity to parameters varies enormously across the
  diffuse weakly-confined state). The curvature correction *amplifies*
  noise rather than geometry.
  - SR at $N = 6, \omega \le 0.01$: $+4$–$36\%$ error.
  - Adam with per-parameter adaptive rates at the same setting: $+0.15$
    and $+0.24\%$.
  - The boundary is *physical*: it coincides with the Wigner crossover
    where the state's sensitivity to single-parameter perturbations
    decouples into orthogonal modes that the Fisher estimate can't
    resolve from 4096 samples.
- **$N{=}20$ aside.** Jastrow-only beats backflow because backflow edge
  features scale $O(N^2 h_{\rm bf})$ and force a hidden-dim / collocation-
  budget cut that destroys gradient quality more than the extra
  expressivity compensates. Architectural expressivity has to be paid for
  in gradient-estimation budget. (Backup slide.)

**Connection.** This slide is the collocation analogue of slide 14. The
two together establish: SR+VMC reaches DMC accuracy where MCMC works;
collocation reaches sub-percent accuracy without any MCMC, in the same
regimes; both pipelines coexist in the code.

**Don't forget.** The single sentence: *we trained an accurate many-body
wavefunction without any MCMC, at sub-percent error against DMC, for
$N \le 12$.*

---

## Slide 16 — Inputs the CTNN uses (attribution)

**On the slide.** `fig_arch_attribution.pdf`, the channel list, the
regime-dependent attribution shifts, the green-box conditioning remark.

**Material.**

- **Input list of the pair branch.**
  - 6 *safe* radial features. With $\tilde r^{\rm soft} = \sqrt{\tilde
    r^2 + \varepsilon^2}$ ($\varepsilon \in [0.15, 0.30]$ trap units):
    $s_1 = \log(1 + (\tilde r^{\rm soft}/\varepsilon)^2)$,
    $s_2 = \tilde r^2 / (\tilde r^2 + \varepsilon^2)$,
    $s_3 = (\tilde r^{\rm soft}/\varepsilon)^2 \exp(-(\tilde r^{\rm
    soft}/\varepsilon)^2)$, and three RBFs $\text{rbf}_g = \exp(-g s_1)$
    for $g \in \{0.25, 1, 4\}$.
  - Raw $(\Delta x_{ij}, \Delta y_{ij})$.
  - One-body positions $(x_i, y_i)$.
  - Spin indicator (same / opposite spin).
- **Safety property.** For each safe feature $s_k$, $ds_k / d\tilde r \to
  0$ as $\tilde r \to 0$. This guarantees that derivatives of the pair
  branch do not inject a $1/\tilde r$ term into $\nabla \log|\Psi|$ or
  $\Delta \log|\Psi|$ — those terms would corrupt the kinetic energy at
  coalescence.
- **Attribution analysis.** Probe the trained network for the contribution
  of each input channel to the output of the Jastrow $W_\theta$. Numbers
  on the slide:
  - $\omega = 1.0$: $|r|$ carries $40.5\%$; spin moderate; raw $(\Delta x,
    \Delta y)$ low.
  - $\omega = 0.1$: spin rises to $51.4\%$.
  - $\omega = 0.001$: spin dominates at $71.6\%$ (Wigner).
- **The raw $(\Delta x, \Delta y)$ channels are always suppressed.** Even
  though we feed them, the network learns to ignore them. They are not
  safe at coalescence; the network prefers well-conditioned radial
  features. *We did not have to tell it.*

### Inputs as a conditioning device

- Redundant inputs are not waste. The network has a menu of bounded,
  informative, mutually overlapping channels and can choose whichever is
  both safe and physically informative in each regime.
- The optimizer migrates onto well-conditioned channels because the NTK
  favors them (they have bounded gradients in parameter space). This is
  exactly the slide-12 Paper-B principle on the input side: well-conditioned
  inputs lower $\kappa$ along the trajectory and let the network find the
  manifold faster.
- Question to anticipate: *are the regime shifts in the inputs used
  actually because of gradient stabilities, or are they regime-shift
  focus?* Honest answer: both. The shift radial$\to$spin tracks the
  physical Wigner crossover (regime-shift focus). The *consistent
  suppression* of raw $(\Delta x, \Delta y)$ across regimes is purely
  gradient-stability (the channel has unbounded derivatives at $r = 0$).
  We have not done a dedicated experiment that decouples the two effects
  — i.e. an ablation that makes the unsafe channel *artificially* safe
  (e.g. multiply by a learned mask vanishing at $r = 0$) and re-runs
  attribution. If asked, propose this as natural follow-up work.

**Don't forget.** This slide is the physics validation of the
architecture, *not* just a learning diagnostic. The attribution shifts
*track the Wigner crossover*.

---

## Slide 17 — Intrinsic dimension: the manifold the network finds

**On the slide.** Effective-rank definition, the table of $r_{\rm eff}$,
$|\rho|$, leading PC branch across $(N, \omega)$, the bullets, the
backflow comparison.

**Material.**

### What is $Z$, and what exactly are we doing PCA *of*? (precise)

- **$Z$ is the correlator's internal feature vector** — the thing fed into
  the readout head $\rho$ just before it outputs the scalar log-amplitude
  correction. It is the concatenation of the three branch outputs:
  $$ Z \;=\; \bar h^\phi \,\big\|\, \bar h^\psi \,\big\|\, \mathbf g, $$
  where $\bar h^\phi$ is the particle-branch embedding *averaged over the
  $N$ particles*, $\bar h^\psi$ is the pair-branch embedding *averaged over
  the $\binom N2$ pairs*, and $\mathbf g$ is the small vector of global
  statistics. So $Z$ is one fixed-length vector *per electron
  configuration* $R$ — the network's compressed summary of that
  configuration, $Z = Z(R) \in \mathbb R^{D}$ (D a few tens).
- **The dataset we PCA.** Draw a batch of $B$ configurations $R^{(1)},
  \dots, R^{(B)} \sim |\Psi|^2$. Push each through the network to get
  $Z(R^{(b)})$. Stack them into a $B \times D$ matrix. *That* matrix is what
  we do PCA on. So we are asking: **as the electrons move around (sampled
  from the physical density), in how many independent directions does the
  network's internal representation $Z$ actually move?** A small answer
  means the correlator, despite living in $\mathbb R^D$, only ever varies
  along a handful of directions — a low-dimensional manifold.
- **Effective rank.** Centre the $B\times D$ matrix, form the covariance
  $\mathrm{Cov}(Z)$, take its eigenvalues $\lambda_1 \ge \lambda_2 \ge
  \cdots \ge 0$ (these are the variances along the principal components),
  normalise $p_i = \lambda_i / \sum_j \lambda_j$, and set
  $$ r_{\rm eff}(Z) = \exp\!\Big(-\sum_i p_i \log p_i\Big). $$
  It is the *exponentiated entropy of the normalized eigenvalue spectrum* —
  a **soft count of how many directions carry real variance**. If one
  eigenvalue dominates, $r_{\rm eff}\to 1$; if $k$ eigenvalues are equal and
  the rest zero, $r_{\rm eff}=k$. (Equivalent in terms of the data matrix's
  singular values $s_i$ via $\lambda_i = s_i^2/(B-1)$ — earlier I wrote it
  with singular values; the eigenvalue form is the cleaner statement and is
  what the slide now shows.)
- **$|\rho|$** in the table is the correlation between the head's scalar
  output and its reconstruction from PC1 alone — "how much of what the head
  computes is captured by the single leading direction." **"PC1 leader"** is
  which branch ($\phi$, $\psi$, or $\mathbf g$) contributes most of PC1's
  variance.

- **Correlator manifold** ($Z = \bar h^\phi \| \bar h^\psi \| \mathbf g$):
  - $r_{\rm eff}(Z) \le 3.1$ across the *whole* grid.
  - At $N = 2$, $\omega = 10^{-3}$: $r_{\rm eff} \approx 1.00$, head/PC1
    correlation $0.998$.
  - For $N \ge 6$: head reconstructed from PC1 alone with $|\rho| > 0.94$.
- **What the leading axes encode** (from linear probes):
  - For $N = 2$: leading PCs predict mean radius and its variance
    essentially perfectly ($R^2 \approx 1$).
  - For $N \in \{6, 12\}$: leading PCs track global size with $R^2 \approx
    1$ on mean radius, $R^2 \in [0.4, 0.85]$ on radial fluctuations.
  - Shell contrast and angular order are only weakly linearly encoded;
    they require explicit angular registration.
- **The $\phi \leftrightarrow \psi$ transition.** At $\omega \ge 0.1$, $N
  \ge 6$: $\psi$ (pair) carries 60–73 % of PC1. At $\omega \le 0.01$, $N
  \ge 12$: shifts to $\phi$ (orbital) or $\mathbf g$ (global density);
  $\psi < 15\%$. Sharpest at $N = 12$: $\psi$ goes from 0.09 at $\omega =
  0.01$ to 0.63 at $\omega = 0.1$.
- **Scaling.** $r_{\rm eff}$ stays $O(1)$ as $N$ grows from 2 to 20.
  The physics is genuinely low-dimensional regardless of the ambient $d$.
  The dimensional reduction is *not* an artifact of small $N$; it is
  the manifold the architecture finds.
- **Backflow geometry — separate object.** $r_{\rm eff}(\Delta x) \approx
  2N - 2$ (full geometric rank, modulo CoM conservation), so the
  displacement field is *high*-rank. But CKA between correlator features
  with-and-without backflow is $> 0.98$ everywhere — the backflow acts
  in the *orthogonal complement* of the correlator's feature manifold.
  In the deep Wigner regime, backflow energy effect drops to $\sim 10^{-7}$
  Ha; CKA $\approx 1$. *Backflow switches off in deep Wigner.*

### Can we run a CTNN-side dimensionality measurement now?

- The thing we currently have is the *post-cell* feature map $Z = \bar
  h^\phi \| \bar h^\psi \| \mathbf g$ — pooled, scalar per particle per
  dimension. That gives the clean rank-3 table.
- The CTNN's *message* tensor $\mathbf m_{ij}$ is higher-rank: it lives in
  $\mathbb R^{B \times N \times N \times d_h}$. PCA on a $B \times (N^2
  d_h)$ flattened matrix would mix the per-pair structure into a single
  rank estimate. Two options if asked to expand the analysis:
  - *Per-pair PCA*: SVD of $\mathbf m_{ij}$ over $B$ for each $(i,j)$;
    report the average $r_{\rm eff}$ across pairs. Measures the per-edge
    feature richness.
  - *Per-particle aggregated message PCA*: SVD of $\mathbf m_i = \sum_j
    \mathbf m_{ij}$ over $B$ per $i$, average over $i$. Measures the
    effective signal each particle receives.
- We have not run either, but the architecture diagnostics scripts could
  do it without retraining (the messages are intermediate activations,
  not stored). If asked, propose this as a follow-up; the result would
  bear directly on whether one MP round saturates the manifold or whether
  $k = 2$ would buy something measurable.

**Connection.** This is the most direct empirical answer to the
*intrinsic-dimensionality* question raised by Paper A Q3. We do not have
a *theorem* that $r_{\rm eff}$ is the right intrinsic dimension of the
physical manifold, but we have an *operationally robust* one: the network
discovers ≤ 3 dimensions, the head is reconstructable from those, and the
linear probes recover physical observables from those same dimensions.

**Don't forget.** The two-mode story of the trained ansatz:
- *Correlator*: low-rank ($\le 3$), global, regime-dependent leading axis,
  encodes order parameters.
- *Backflow*: high-rank ($\sim 2N$), local, near-identity ODE flow,
  encodes nodal geometry.
These are not redundant. They are complementary, and their separation
is why the ansatz works across the entire crossover.

---

## Slide 18 — Three-body sensitivity

**On the slide.** The definition of the intra-bin variance ratio, the
$N = 6$ data $r(1.0) = 2.03$, $r(0.1) = 2.93$, $r(0.001) = 1.05$, the
interpretation, the "codimension-in-pairwise" green box.

### First, the question this measurement answers

*"When the network decides how to correlate particles $i$ and $j$, does it
look only at how far apart $i$ and $j$ are — or does it also look at where
the **other** electrons are?"*

- If it only ever looks at the pair distance $r_{ij}$, the correlation is
  **two-body** (pairwise): a sum of functions of one distance at a time.
  That is what a classical Jastrow does.
- If the answer for the $(i,j)$ pair *changes* when you move a **third**
  electron $k$ (even with $r_{ij}$ held fixed), the network is using
  **three-body** (beyond-pairwise) information. That is the thing the
  message-passing cell can do and a pairwise factor cannot.

The three-body sensitivity ratio is just a number that measures *how much*
of the latter is happening, in each physical regime.

### Why the CTNN cell *can* be three-body (one line of architecture)

A one-round message-passing update for particle $i$ is
$$ \delta\tilde x_i = \psi_\beta\Big(\tilde x_i \;\Big\|\;
   \underbrace{\textstyle\sum_{j \ne i}\phi_\beta(\tilde x_i, \tilde x_j,
   \mathbf r_{ij}, \ldots)}_{\text{sum of messages from all neighbours}}\Big). $$
The update to $i$ depends on a **sum over all other particles**. So if you
hold $r_{ij}$ fixed but move a third particle $k$, the message
$\phi_\beta(i,k,\ldots)$ changes, the sum changes, and the output for $i$
changes. The cell's output for one pair is contaminated (in the good sense)
by the rest of the cloud — that contamination *is* the three-body content.
A strictly pairwise factor $\sum_{i<j} f(r_{ij})$ has, by construction, no
such dependence: fix $r_{ij}$ and its contribution is frozen.

### The calculation, concretely (step by step)

1. **Sample** many configurations $R$ (all $N$ electron positions) from
   $|\Psi|^2$.
2. **Bin by pair distance.** Group the configurations into bins so that
   within one bin every configuration has *the same* value of a chosen pair
   distance $r_{ij}$ (say all configs with $r_{ij} \approx 1.3$ go in one
   bin). Inside a bin, the pair distance is held fixed *by construction*;
   the only thing that varies is *where the other electrons are*.
3. **Look at the cell's output inside a bin.**
   - A purely pairwise network would output the *same* number for every
     configuration in the bin (it only sees $r_{ij}$, which is fixed). So
     its **variance within the bin is zero** — that is the control / null
     model.
   - The CTNN cell can output *different* numbers within the bin, because it
     also sees the other electrons. Its **intra-bin variance is nonzero**
     exactly to the extent it uses beyond-pairwise information.
4. **Take the ratio.**
   $$ r(\omega) = \frac{\mathrm{Var}_{\rm intra}[\text{CTNN cell output}]}
   {\mathrm{Var}_{\rm intra}[\text{pairwise baseline}]}. $$
   (The denominator is a small reference number — the residual variance of
   a matched pairwise model, essentially the noise floor — so the ratio is
   normalised to "$1 =$ no more than pairwise".)
   - $r \approx 1$ → the cell output at fixed $r_{ij}$ barely moves when the
     rest of the cloud moves → **the network is behaving pairwise.**
   - $r > 1$ → it moves a lot → **the network is genuinely using
     three-body structure.**

So the whole measurement is: *"hold the pair distance fixed, wiggle the
third particle, and see whether the network's answer wiggles too."* The
ratio is how much it wiggles, relative to a pairwise model that cannot
wiggle at all.

### What the numbers say (N=6) and the implication

- $r(1.0) = 2.03$ — moderate confinement (correlated fluid): the cell does
  use three-body content; its output roughly doubles in variance over
  pairwise.
- $r(0.1) = 2.93$ — intermediate regime: the **most** three-body content of
  the three. This is the regime where the cell contributes the most energy
  too (the $+30.5\%$ ablation), and where the latent manifold is richest
  ($r_{\rm eff}$ peaks). The physics genuinely needs to know about triplets:
  whether electron $i$'s correlation with $j$ should change because a third
  electron $k$ is nearby screening or crowding.
- $r(0.001) = 1.05$ — deep Wigner crystal: three-body content essentially
  **vanishes**. Once the electrons freeze onto a rigid lattice, knowing two
  positions fixes the rest by the crystal geometry — there is no
  *independent* third-particle freedom left for the network to exploit. The
  correlation is effectively pairwise (each electron just repels its fixed
  neighbours).

**The implication.** The message-passing machinery — the only part of the
architecture that gives beyond-pairwise capacity — is **not always on**.
It earns its keep at moderate correlation ($\omega \gtrsim 0.1$) and goes
dormant in the deep Wigner limit. This is *consistent and mutually
reinforcing* with the other two diagnostics:
- the **ablation** (removing message passing) costs $+30.5\%$ at $\omega =
  0.1$ but only $+11.8\%$ at $\omega = 0.001$ — biggest where $r$ is biggest;
- and the energy cost is **entirely kinetic** — the cell is reshaping nodal
  geometry, which is where three-body correlation would act.

So three independent measurements (sensitivity ratio, ablation energy,
kinetic-only signature) all point at the same statement: *beyond-pairwise
structure matters at moderate correlation and disappears at the Wigner
lattice.* That coherence is the real result.

### The codimension-in-pairwise observation (green box)

- Let $\mathcal F_k$ be the class of $k$-round permutation-invariant
  message-passing networks. $\mathcal F_0$ is the strictly-pairwise
  sub-class.
- Claim (data-supported, not yet a theorem): $r(\omega) \to 1$ when the
  optimal correlator in $\mathcal F_k$ lies in $\mathcal F_0$;
  $r(\omega) > 1$ when it does not.
- *What would turn this into a theorem*:
  - Precise statement of $\mathcal F_k$ for our specific message function
    and soft-core regularity.
  - Lower bound: intra-bin variance bounded below by $L^2$ distance from
    optimal $\mathcal F_k$ representative to its $\mathcal F_0$ projection.
  - Explicit construction of the optimal pairwise correlator in the
    Wigner regime (likely a Gutzwiller-like factor on lattice sites).
  - None of these is hard; I have not done it.
- This is the talk's one original mathematical observation. *The
  variance ratio measures the codimension of the physical solution
  inside the architectural function class.*

**Connection.** Operationalizes Paper A's GNN-expressivity claim on a
physical problem. The cell carries beyond-pairwise information *exactly*
where the physics needs it ($\omega \ge 0.1$) and disengages where the
physics collapses to pairwise (deep Wigner). This is what
$k$-round-message-passing was *for*, validated on a problem where we know
the physical answer.

**Don't forget.** If you have time, give them the codimension observation
as your one original contribution. If you don't, just give the data and
the two-line interpretation.

---

## Slide 19 — Synthesis

**On the slide.** Three bullets uniting Paper A + Paper B + our work,
the take-away green box.

**Material.**

- **Paper A**: the manifold *exists* — Barron, depth, compositionality,
  symmetry break the curse; architecture encodes the symmetry.
- **Paper B**: the manifold is *reachable* iff $\kappa(L^* L \circ TT^*)$
  is controlled — almost every successful PINN technique is
  preconditioning.
- **Our PINN** sits at the intersection: conditioning (cusp, soft
  features, trap units, REINFORCE, SR/Adam regime split) brings $r_{\rm
  eff} \le 3$ within reach for $d = 40$.

**Take-away in one sentence (green box).** *The many-body wavefunction
lives on a 1–3 dimensional manifold that a symmetry-equivariant,
well-conditioned neural network can reach with $\sim 3 \times 10^3$ MCMC
samples per SR step — orders of magnitude better than basis blow-up
implies.*

**Closing line if you have a moment.** *"Both papers, even when stated in
pure mathematics, are answering one question: when is high-dimensional
learning actually feasible? The answer is: when the target has structure,
and when the optimizer can see the structure. We built a worked example
for the many-body Schrödinger equation, and every step of that example is
one of these papers' predictions made concrete."*

Invite a question: *"happy to go deeper on any of these."*

---

## Backup slides

These exist for *if asked*. Brief notes.

### Backup A — NTK proof sketch and the linearization

On the slide: the chain-rule derivation $\dot f_t(x) = -\sum_{x'}
\Theta_t(x, x') (f_t(x') - y(x'))$, Jacot's frozen-kernel limit, the
identification with SR.

Material already in slide 7 notes; the slide repeats the key formulas
verbatim so you can write them on the board if asked.

### Backup B — Barron proof sketch

On the slide: the inverse-Fourier identity, the re-expression as an
expectation against $\mu_g$, Monte Carlo over $n$ samples, Bienaymé's
identity giving $L^2$ error $\le 2\pi C_g / \sqrt n$, single-neuron
approximation of $\Gamma$. Green-box punchline: *"the neurons are Monte
Carlo samples in frequency space."*

Material in the slide 9 notes — that's the depth version.

### Backup C — Design choices → preconditioners

On the slide: a table mapping our techniques (trap units, soft-core
features, analytic cusp, variational/REINFORCE, SR) to what they
condition and the $\kappa$ effect. Plus the local-energy + REINFORCE
formula.

Material: each row is one paragraph from the slide-12 notes (preconditioning
principle section). The slide is the lookup table; you talk it.

### Backup D — 2D cusp + log-divergent squared spike

On the slide: spin-resolved cusp slopes, $E_L \supset -\delta a / r$,
$\int_{r < \varepsilon}(1/r)^2 d^2 \mathbf r = +\infty$.

Material. The 2D log-divergence is the deepest single technical reason
our cusp engineering matters so much. In 3D the integral is finite
($1/r^2 \cdot r^2 dr$ has measure-zero singular point); in 2D it is
log-divergent ($1/r^2 \cdot r \, dr = dr/r$). So in 2D quantum dots, any
*finite* cusp-slope error gives an integrand whose square has infinite
expectation under the sampling measure near coalescence. Heavy-tailed
local energy, broken variance estimator, broken everything. Our analytic
cusp fixes this exactly.

### Backup E — Backflow as a force-aligned object across the Wigner crossover

On the slide: median cosines of $\Delta x$ with trap force, Coulomb
force, total force for $N = 6$. The sign flip at $\omega = 0.001$.

Material. In the correlated-fluid regime ($\omega \ge 0.1$), the backflow
aligns with the trap force ($\cos \approx +0.97$) and opposes the
Coulomb repulsion ($\cos \approx -0.90$): it is a *trap-restoring* mode
that pushes electrons toward configurations preferred by the confining
potential, correcting over-repulsion of an imperfect Coulomb node
geometry.

At the Wigner crossover ($\omega = 0.001$), both signs flip completely:
$\cos(\Delta x, F^{\rm trap}) \approx -0.83$, $\cos(\Delta x, F^{\rm Coul})
\approx +0.83$. The backflow now follows Coulomb and opposes the trap —
a *lattice-correction mode*. The Wigner crystal has discrete rotational
symmetry that the trap does not; the learned backflow adjusts the nodal
geometry to accommodate the lattice structure against the trap's
isotropic restoring force.

This sign reversal coincides precisely with the Lindemann melting /
topology collapse signatures of the Wigner crossover identified by the
structural diagnostics. The backflow's *learned alignment target*
literally changes from trap to interaction as you cross the phase
boundary — a striking architectural validation of the physical regime.

Worth offering if the conversation turns to "what does backflow actually
*do*?" — this is the most physically vivid answer.

### Backup F — Sharp PINN–CTNN coupling transition at $\omega \approx 0.1$

On the slide: the small table of $\langle \cos(\nabla_x f, \Delta x)
\rangle$ and $|\delta f| / \sigma_f$ across $N$ and $\omega$.

Material. Below $\omega \approx 0.1$ ($N = 6, 12$): $\langle \cos \rangle
\sim 0.1$–$0.2$, $|\delta f|/\sigma_f < 0.07$, alignment fraction well
below 1. The CTNN operates in a subspace *orthogonal* to the PINN
gradient — modifies the wavefunction in directions the correlator output
$f$ is insensitive to. Its job here is to adjust orbital positions
through the determinant, not to perturb the correlator.

Above $\omega \approx 0.1$: $\langle \cos \rangle$ jumps to 0.5–0.8,
alignment fraction = 1.0 for $N \ge 6$, $|\delta f|/\sigma_f$ reaches
25–69 %. Cooperative refinement — the CTNN pushes configurations along
directions the PINN is maximally sensitive to, systematically correcting
correlator residual errors.

The transition coincides with the physical crossover in $\Gamma = V_{\rm
int}/T$ from $\Gamma > 10$ (interaction-dominated) to $\Gamma \sim 3$–$6$
(moderate correlation). The CTNN switches *operational mode* with the
physics — independent in strong correlation, cooperative in moderate
correlation.

At $N = 12, \omega = 0.001$: $\langle \cos \rangle = -0.48$ (anti-aligned).
This is the most extreme case. The CTNN actively pushes *against* the
PINN gradient — consistent with the lattice-correction role from the
force-alignment data above.

Worth offering as the second-level architectural story: the CTNN is not
"on" or "off", it has two qualitatively different operational regimes,
and the boundary is the same physical crossover that controls everything
else.

### Backup G — $N = 20$: Jastrow-only beats backflow

On the slide: the comparison $+1.45\%$ (J-only) vs $+18\%$ (BF) at $N =
20$, $\omega = 1.0$, with the $O(N^2 h_{\rm bf})$ memory-scaling
explanation.

Material. The BF edge features scale as $O(N^2 \times h_{\rm bf})$ — at
$N = 20$ this dominates memory and forces a severe cut in hidden
dimension ($h_{\rm bf} = 64$ is already small), collocation budget
($n_{\rm coll}$ down to a few thousand), or oversampling factor. All
three cuts degrade gradient quality. The Jastrow-only ansatz is much
smaller, so it can use the full collocation budget at full
$8 \times$ oversampling.

The implication is *not* "BF is bad". It is: *architectural expressivity
must be paid for in gradient-estimation budget*. The SR / ESS analysis
predicts exactly where the trade reverses — the BF wins the expressivity
argument but loses the budget argument at fixed compute.

A genuine prediction of the gradient-quality / expressivity tradeoff
from the conditioning analysis, on a problem where we can measure both
sides. Worth offering if the discussion turns to scaling.

---

## Practical reminders before walking in

- **Names**: Berner / Grohs / Kutyniok / Petersen (A). De Ryck / Mishra
  (B). Barron. Jacot (NTK). Mhaskar–Poggio (compositionality).
  Carleo–Troyer (NQS). Pederiva (DMC reference for our problem).
  Haas (Deep-Sets-needs-correlation lemma).
- **Two equations on the board if needed**: local energy
  $E_L = -\frac12(\Delta \log|\Psi| + \|\nabla \log|\Psi|\|^2) + V$;
  $\kappa(A) = \kappa(L^* L \circ T T^*)$.
- **Three percentages**: 22 / 30 / 12 for the CTNN ablation at $\omega
  = 1.0 / 0.1 / 0.001$.
- **Three numbers from intrinsic dim**: $r_{\rm eff} \le 3$ everywhere;
  $|\rho| > 0.94$ for $N \ge 6$; $r(\omega) = 1.05, 2.03, 2.93$ at
  $\omega = 0.001, 1.0, 0.1$ for the three-body sensitivity ratio.
- **The single best sentence**: *"Most successful PINN tricks are
  preconditioning, and the strongest version of that statement we have
  is our negative-result table — 17 parameter-space interventions all
  fail at $0.32\%$, only changing the form (REINFORCE / variational +
  SR) reaches $< 0.01\%$."*
- **Color navigation on slides**: blue = Paper A, red = Paper B, green
  = us.
- **Breathe.** You wrote this thesis. They are asking you to teach them.
