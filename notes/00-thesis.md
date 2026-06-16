# RecursiveNe — Thesis

## 1. LLMs are trained in reverse
An LLM ingests the *output* of human cognition — language — under massive supervision, and
backs into a world model that is implicit, shaky, and never grounded in action. A child does
the opposite: it learns a predictive **world model** from its own self-supervised sensorimotor
stream, with no labels, spending samples only where they buy competence; **language arrives
last**, as a compression/communication layer over the already-grounded model. RecursiveNe is
built in the evolutionary order — world model first, symbols last — and engineered to get
**cheaper at learning over time** rather than larger.

This is not a lone intuition. It is LeCun's JEPA critique (predict in representation space, not
token space), Friston's free-energy principle (act and learn to minimize surprise about your
own sensory stream), Schmidhuber's artificial curiosity (reward = *learning progress*), and the
Dreamer line (learn a world model, then imagine in it). RecursiveNe fuses them into one minimal
seed and a loop that improves the seed.

## 2. Least-training-intensive core
The core update is **online, closed-form, backprop-free**: one observation → exactly one
bounded-cost update (RFF + recursive least squares). No epochs, no replay buffer, no minibatch,
no global backward pass. The *prior prediction error* is simultaneously the learning signal and
the intrinsic-reward (surprise) signal — one quantity computed once. The model's entire
parameter budget is one number, `D`, and the whole system is built to drive `D` down while
holding competence: the **race to 0**. (Predictive coding is the biologically-plausible
generalization; see `knowledge/local-learning-rules.md`.)

## 3. Curiosity that survives the noisy TV
Sample efficiency is won by *what you choose to experience*. Naive novelty (chase the highest
prediction error) is trapped forever by an unlearnable noise source — the "noisy TV". The fix is
**learning progress**: pursue error that is actually *reducible*. RecursiveNe's policy targets
**the worst still-learnable activity** — novelty says what is worth fixing, learnability says
which of those is fixable (vs noise). Measured result: in a world that is 67% unlearnable noise,
the learning-progress learner reaches competence in fewer samples than random *and* spends ~13%
of its budget on noise vs naive curiosity's ~95%. (The deep, unfinished problem — separating a
slow-but-learnable signal from high-amplitude noise over a short window — is exactly what the
`knowledge/robust-learning-progress.md` feed targets.)

## 4. Weak RSI is not new; strong RSI is the frontier
"Computers improving computers" is as old as compilers. That is **weak RSI**: a *frozen*
optimizer improves an artifact, and plateaus at the optimizer's ceiling. **Strong RSI** is the
human/cultural pattern — improve *the means of improvement itself*. The analogues all share one
structure: the improver is **inside** the thing being improved.

| Domain | Weak (improve content) | Strong (improve the improver) |
|---|---|---|
| Evolution | select fitter organisms | evolve *evolvability* (the genetic code, modularity, sex) |
| Culture | learn facts | invent language, writing, math — tools that build better tools |
| Science | gather data | invent the scientific method, then revise it *with itself* |
| Compilers | recompile faster | invent a better *language* to express the compiler in |
| ML | tune weights | meta-learn the learning rule; grow the library of primitives |

RecursiveNe makes this concrete as a **three-level fixed point** (object → meta → meta-meta) under
a fixed objective. Perfect RSI improves **both the model and the harness**, and the harness that
improves the harness, with **no privileged frozen outer loop** — only the *objective* and a
*held-out invariant* stay fixed, as the ruler the system cannot bend toward itself.

## 5. A fixed test saturates and then lies
Ace the test and you need a new test. Any fixed benchmark, once saturated, stops driving
improvement and starts inviting Goodhart. So the **test-maker must be inside the loop**: a
problem generator that, at the mastered frontier, invents problems gated to be **novel +
previously-unsolvable + now-solvable + non-forgetting** (POWERPLAY / POET / novelty search), and
mints **abstractions** that expand what can even be represented (the cultural ratchet;
DreamCoder-style library growth). Progress is then measured by **non-saturating** metrics —
repertoire growth, hardest-solved complexity, transfer — not a frozen number. It is
learning-progress all the way up: sample → task → problem → abstraction → improvement-operator.

## 6. Honest walls
- **Löbian obstacle**: a formal system cannot fully prove a successor as strong as itself sound
  (Gödel II applied to self-trust). RecursiveNe therefore certifies successors **empirically** —
  held-out frontier performance — not by proof. The Gödel-machine ideal (only provably-beneficial
  self-rewrites) is the limiting case; we use a sandbox + held-out invariant as the practical one.
- **No free lunch**: improvement is always relative to a problem distribution. Strong RSI escapes
  this by **co-evolving the distribution with the solver** (the generator), so there is always a
  next rung — not by a universal optimizer.
- **Wireheading / Goodhart**: the defense is structural — the objective and the invariant are
  protected, the reward is measured on unseen worlds, and the kill switch is outside the editable
  surface. The system cannot "improve" by redefining success.

See `01-roadmap.md` for how the L0 seed grows toward this without retraining from scratch, and
`citations.md` for sources.
