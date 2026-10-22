# main.tex patches: the pseudo-action is drawn uniformly (p = 1/K)

The pseudo-action of Eq. (12) is kept, and drawn round-robin -- that is,
uniformly over the arms.  Setting `p = 1/K` makes Eq. (12) exactly that:

    phi(a = a_tau)  = p           = 1/K
    phi(a != a_tau) = (1-p)/(K-1) = 1/K

so `1/phi = K`, and **Eq. (13) is correct exactly as written**.  Per-round
unbiasedness survives, the matching event and the redraw cap survive, and the
proofs keep their shape.  Far less changes than a pure round-robin would need.

Nothing in Section 3 or the appendix has to be deleted.  Below are the four
places where the value of `p` is asserted or used.

---------------------------------------------------------------------------
## A. Eq. (12) -- the range of p

### before
```latex
where coupling probability $p \in (\tfrac12, 1)$, which is bounded away from zero for every arm.
```

### after
```latex
where the coupling probability is set to $p=1/K$, at which \cref{eq:pseudo_policy}
is the uniform draw: $\phi_{a,\tau}=1/K$ for every arm, whether or not it is the
one that was pulled.  This is the propensity that a round-robin visit carries,
and it is bounded away from zero for every arm.
```

Then Eq. (13)'s $1/\phi_{a,\tau}$ equals $K$ and needs no change, and the
sentence after it --- `Since $\EE[\II(a_\tau=a)]=\phi_{a,\tau}$ ...` --- holds as
written, because $\tilde a_\tau$ is genuinely random.

---------------------------------------------------------------------------
## B. Delete the red note under Eq. (13)

```latex
\sj{실험 진행시 IPW=1/p로 진행할 때 t가 터집니다. 오히려 $\psi$를 1/K로 진행했을 때 IPW=K가 되면서서 오차 없이 빠르게 t가 멈춥니다..}
```
This is now the stated design rather than an open question.  Appendix
`app:experiments` item (i) records the measurement.

---------------------------------------------------------------------------
## C. Algorithm 1 -- INPUT line

### before
```latex
\STATE \textbf{INPUT:} observed features $\{\xb_{a}\}_{a\in[K]}$, coupling probability $p\in(\tfrac12,1)$, confidence parameter $\delta>0$.
```

### after
```latex
\STATE \textbf{INPUT:} observed features $\{\xb_{a}\}_{a\in[K]}$, confidence parameter $\delta>0$.
```
$p=1/K$ is determined by $K$, so it is not a parameter the caller supplies.
The body of the algorithm is unchanged.

---------------------------------------------------------------------------
## D. Appendix `app:algorithm` -- the rho_t paragraph

Two sentences assume $p>1/2$ and are false at $p=1/K$.  The paragraph is
otherwise correct; Eq. `eq:rho` itself needs no change.

### before
```latex
As $p>1/2$ we have $\rho_t=O(\log(t/\delta'))$, while the expected number of draws is $1/p<2$ and does not grow with $t$; the cap is a guarantee of termination rather than a cost incurred in typical rounds.
```

### after
```latex
At $p=1/K$ we have $\rho_t=O(K\log(t/\delta'))$, since $\log(1/(1-1/K))\ge1/K$,
while the expected number of draws is $1/p=K$ and does not grow with $t$.  Both
are free: a pseudo-draw consumes no pull, so the cap is a guarantee of
termination rather than a cost paid in rewards.
```

Also add, after the existing sentence about the sweep Gram matrix:
```latex
The same round-robin schedule fixes the weight in \cref{eq:dr_reward}.  Over a
sweep the imputation contributes
$\sum_{\tau\in s}(\Gb-K\hat\xb_{a_\tau}\hat\xb_{a_\tau}^\top)=K\Gb-K\Gb=\zero$
whenever $\check\mub$ is held fixed across the sweep, an identity rather than a
statement in expectation.  At any weight other than $K$ the sweep leaves
$(K-w)\Gb$ behind and the confidence width has to carry it.
```
with $\Gb:=\sum_{a\in[K]}\hat\xb_a\hat\xb_a^\top$ (add
`\providecommand{\Gb}{\mathbf{G}}` to the preamble if `abbe` does not define it).

Note this requires $\check\mub$ to be constant across a sweep; if you want that
stated in the paper rather than left to the implementation, index
\cref{eq:imputation} by the sweep instead of by $\tau$.  Otherwise
`app:experiments` continues to record it as an implementation departure.

---------------------------------------------------------------------------
## E. Unrelated, but still outstanding

- `\cref{thm:lb}` -- no such label; the theorem is
  `thm:sample_complexity_lower_bound`.  Search and replace.
- `\label{app:experiments}` is referenced by Section 4 but never defined.  Add
  `\section{Experiments}\label{app:experiments}\input{experiments}` in the
  appendix, and `\graphicspath{{results/}}` to the preamble.
- `cor:est_error` proof: `\Mcal_t` is defined with `\log(1/p)` where Eq. (15)
  has `\log(1/(1-p))`.  A single draw fails with probability $1-p$, so Eq. (15)
  is right and the proof has a typo.
- `cor:est_error` establishes
  $\sqrt{2(d+1)/t}\,(6\sigma/p\sqrt{\log(t^2/\delta)}+3/2)$, which is not the
  width Eq. (15) is run with.  The experiment uses
  $\sigma\sqrt{2K\log(2K/\delta')/N}$; the two differ by 2.4x to 3.3x over
  $t\in[20,10^4]$ and, more importantly, in their dependence on $K$ --- the
  corollary carries $d$ where the setting calls for $K$, for the same reason
  `thm:sample_complexity_lower_bound` does.  At $p=1/K$ the corollary's
  $6\sigma/p$ becomes $6\sigma K$, which is the right order.
