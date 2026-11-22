"""Helper code for A5 - Neural Networks."""

import numpy as np


class Classifier:
    def classify(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def penalty(self) -> float:
        return 0.0

    def cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """What training minimizes: loss plus penalty."""
        return self.loss(X, y) + self.penalty()

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """The fraction of X classified correctly."""
        return np.mean(self.classify(X) == y)


def softmax(S: np.ndarray) -> np.ndarray:
    """Scores (n, c) -> probabilities (n, c). Subtracting each row's max
    changes nothing, since softmax(s + a) = softmax(s), and keeps exp from
    overflowing."""
    E = np.exp(S - S.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

#def svm(S: np.ndarray) -> np.ndarray:


class SoftmaxClassifier(Classifier):
    """Any model whose scores, model(X) of shape (n, c), become class
    probabilities by softmax; labels 0, ..., c-1."""

    def classify(self, X: np.ndarray) -> np.ndarray:
        """The class with the biggest score, shape (n,)."""
        return np.argmax(self(X), axis=1)

    def probs(self, X: np.ndarray) -> np.ndarray:
        """softmax of the scores, shape (n, c)."""
        return softmax(self(X))

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean negative log probability of the correct class (A2's NLL)."""
        P = self.probs(X)
        return -np.mean(np.log(P[np.arange(len(y)), y]))


# ===========================================================================
# A4's linear classifier, with params as a dict
# ===========================================================================


class LinearClassifier(SoftmaxClassifier):
    """Scores X @ W + b, one column per class."""

    def __init__(self, d: int = 2, c: int = 3, reg: float = 0.0) -> None:
        self.params = {"W": np.zeros((d, c)),
                       "b": np.zeros(c)}
        self.reg = reg
        
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Scores for X, shape (n, d) -> (n, c)."""
        return X @ self.params["W"] + self.params["b"]

    def penalty(self) -> float:
        """L2. b is never penalized."""
        return self.reg * np.sum(self.params["W"] ** 2)

    def grad(self, X: np.ndarray, y: np.ndarray) -> dict:
        """The gradient of cost, penalty included."""
        dS = self.probs(X)
        dS[np.arange(len(y)), y] -= 1
        dS /= len(y)
        return {"W": X.T @ dS + 2 * self.reg * self.params["W"],
                "b": dS.sum(axis=0)}


# ===========================================================================
# Part 1: A two-layer network
# ===========================================================================


class TwoLayerNet(SoftmaxClassifier):
    """Hidden layer H = relu(X @ W1 + b1), then scores H @ W2 + b2, one
    column per class.

    d inputs, h hidden units, c classes.
    """

    def __init__(self, d: int = 2, h: int = 10, c: int = 3, reg: float = 0.0,
                 scale: float = 0.0) -> None:
        # Every weight starts at scale.
        self.params = {
        "W1": np.random.randn(d, h) * np.sqrt(2/d),
        "b1": np.zeros(h),
        "W2": np.random.randn(h, c) * np.sqrt(2/h),
        "b2": np.zeros(c)
    }
        self.reg = reg

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Scores for X, shape (n, d) -> (n, c)."""
        H = np.maximum(0, X @ self.params["W1"] + self.params["b1"])
        return H @ self.params["W2"] + self.params["b2"]

    def penalty(self) -> float:
        """L2 on both weight matrices. b1 and b2 are never penalized."""
        return self.reg * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))

    def one_hot(self, X:np.ndarray, y:np.ndarray, p:np.ndarray):
      c = p.shape[1]
      one_hot = np.eye(c)[y]
      return one_hot

    def grad(self, X: np.ndarray, y: np.ndarray) -> dict:
        p = self.probs(X) 
        # H has shape (n,h)
        H_in = X @ self.params["W1"] + self.params["b1"]
        H_out = np.maximum(0, H_in)

        #shape (n,c)
        y_copy = self.one_hot(X,y,p)
        dLoss_scores = p - y_copy
        # shape (n,h)
        dScores_W2 = H_out
        dScores_b2 = 1
        #shape h,c
        dScores_outh = self.params["W2"]

        # shape(n,h)
        dOuth_inh = np.zeros((H_out.shape[0], H_out.shape[1]))
        dOuth_inh = np.where(H_in > 0, 1, 0)

        # shape (n,d)
        dInh_W1 = X
        dInh_b1 = 1

        #(should be shape h,c)
        dw2 = (dScores_W2.T @ dLoss_scores)/H_out.shape[0]
        dw2 += 2*self.reg * self.params["W2"]
      

        #shape c
        db2 =  np.mean(dLoss_scores, axis = 0)

        #should be shape d,h
        dw1 = (dInh_W1.T @ ((dLoss_scores @ dScores_outh.T) * dOuth_inh))/X.shape[0]
        dw1 += 2*self.reg * self.params["W1"]
        db1 = np.mean(((dLoss_scores @ dScores_outh.T) * dOuth_inh), axis = 0)
        return {"W2": dw2,
                "b2": db2,
                "W1": dw1,
                "b1": db1}

        ## TODO for part 1: the exact gradient of cost, penalty included, as a
        ## dict with one array per name in params, each shaped like that
        ## parameter. Check it with grad_check.


def numerical_grad(model, X: np.ndarray, y: np.ndarray,
                   h: float = 1e-5) -> dict:
    """The derivative of model.cost(X, y) with respect to each parameter in
    model.params, as a dict of arrays shaped like the parameters. Two calls
    to cost per number in the model: use a small model."""
    grads = {}
    for name, original in model.params.items():
        value = np.array(original, dtype=float)
        grad = np.zeros_like(value)
        for i in np.ndindex(value.shape):
            nudge = np.zeros_like(value)
            nudge[i] = h
            model.params[name] = value + nudge
            up = model.cost(X, y)
            model.params[name] = value - nudge
            down = model.cost(X, y)
            grad[i] = (up - down) / (2 * h)
        model.params[name] = original
        grads[name] = grad
    return grads


def grad_check(model, X: np.ndarray, y: np.ndarray, randomize: bool = True,
               seed: int = 0) -> dict:
    """Compare model.grad(X, y) with numerical_grad, one parameter at a time.
    Prints and returns the worst difference relative to the biggest entry of
    either: tiny (around 1e-7 or less) means grad is right.

    randomize=True checks at random parameters, then puts the model's own
    back: at special parameters, like all zeros, a wrong grad can agree with
    numerical_grad. randomize=False checks the model where it is.

    Only the first 10 rows of X and y are used: fast, and with few rows a
    nudge rarely crosses a relu corner, where numerical_grad is wrong.
    """
    saved = model.params
    if randomize:
        rng = np.random.default_rng(seed)
        model.params = {name: rng.standard_normal(np.shape(value))
                        for name, value in saved.items()}
    X, y = X[:10], y[:10]
    analytic = model.grad(X, y)
    numeric = numerical_grad(model, X, y)
    model.params = saved
    worst = {}
    for name in model.params:
        assert np.shape(analytic[name]) == np.shape(numeric[name]), \
            f"grad for {name} has shape {np.shape(analytic[name])}, but {name} has shape {np.shape(numeric[name])}"
        biggest = max(np.abs(analytic[name]).max(), np.abs(numeric[name]).max(), 1e-12)
        worst[name] = np.abs(analytic[name] - numeric[name]).max() / biggest
        print(f"{name}: worst relative difference {worst[name]:.1e}")
    return worst


# ===========================================================================
# Training
# ===========================================================================


class Optimizer:
    """Moves a model's parameters, given their gradients.

    step(model, grads) changes every entry of model.params; grads is a dict
    with the same names, as model.grad returns. An optimizer may remember
    earlier steps, so use a new one for every training run.
    """

    def step(self, model, grads: dict) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Each parameter moves by -lr times its gradient."""

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def step(self, model, grads: dict) -> None:
        for name, value in model.params.items():
            model.params[name] = value - self.lr * grads[name]


def train(model, optimizer, X, y, X_val, y_val, epochs=10, batch_size=100, seed=0):
    rng = np.random.default_rng(seed)
    history = {"batch cost": [], "train loss": [], "val loss": [],
               "train accuracy": [], "val accuracy": []}

    def record_epoch():
        history["train loss"].append(float(model.loss(X, y)))
        history["val loss"].append(float(model.loss(X_val, y_val)))
        history["train accuracy"].append(float(model.accuracy(X, y)))
        history["val accuracy"].append(float(model.accuracy(X_val, y_val)))

    record_epoch()
    for epoch in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            history["batch cost"].append(float(model.cost(X[batch], y[batch])))
            optimizer.step(model, model.grad(X[batch], y[batch]))
        record_epoch()
    return history


# ===========================================================================
# Part 2: Optimizers
# ===========================================================================


class Surface:
    """A model that is nothing but a cost of two parameters, θ1 and θ2, for
    watching optimizers walk.

    Like a real model, it has data X and y, and its cost is a mean over the
    data: f(θ1, θ2, X, y) gives one cost per data point, and cost is their
    mean. So a batch of rows is a minibatch, exactly as in train, and each
    data point has a cost surface of its own. A surface with no data keeps
    X = None and has f return the cost itself.

    The surfaces come from SGD Lab: its Copy as Python button gives a
    subclass with the data and f filled in. f uses numpy functions, so that
    it works when θ1 and θ2 are grids of numbers (that is how plotting draws
    it). grad is numerical_grad, so there is no gradient to write.
    """

    X = None
    y = None
    start = (0.0, 0.0)
    window = (-3.0, 3.0, -3.0, 3.0)        # θ1 from, θ1 to, θ2 from, θ2 to

    def __init__(self, θ1: float | None = None, θ2: float | None = None) -> None:
        if θ1 is None:
            θ1, θ2 = self.start
        self.params = {"θ1": np.float64(θ1), "θ2": np.float64(θ2)}

    def f(self, θ1, θ2, X, y):
        raise NotImplementedError

    def at(self, θ1, θ2, X=None, y=None):
        """The cost at θ1, θ2 (numbers, or grids of numbers), as a mean over
        the rows of X and y: all of the data if X is None."""
        if X is None:
            X, y = self.X, self.y
        if X is None:
            return self.f(θ1, θ2, None, None)
        θ1, θ2 = np.asarray(θ1)[..., None], np.asarray(θ2)[..., None]
        return np.mean(self.f(θ1, θ2, X, y), axis=-1)

    def cost(self, X=None, y=None) -> float:
        return self.at(self.params["θ1"], self.params["θ2"], X, y)

    def grad(self, X=None, y=None) -> dict:
        return numerical_grad(self, X, y)

class AdamW(Optimizer):
  def __init__(self, lr: float, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weight_decay = 0.01) -> None:
          self.lr = lr
          self.beta1 = beta1
          self.beta2 = beta2
          self.epsilon = epsilon
          self.v = {}
          self.q = {}
          self.t = 0
          self.weight_decay = weight_decay
  def step(self, model, grads:dict) -> None:
    self.t += 1
    for name, value in model.params.items():
        if name not in self.q:
          self.v[name] = np.zeros_like(value)
          self.q[name] = np.zeros_like(value)
        self.v[name] = self.v[name] * self.beta1 + grads[name] * (1-self.beta1)
        self.q[name] = self.q[name] * self.beta2 + grads[name]**2 * (1-self.beta2)

        first_unbias = self.v[name]/ (1-self.beta1**self.t)
        second_unbias = self.q[name] / (1-self.beta2**self.t)
        if (name == "W1" or name == "W2"):
          model.params[name] = model.params[name] - self.lr * ((first_unbias/ (np.sqrt(second_unbias) + self.epsilon)) + self.weight_decay * model.params[name])

        else:
           model.params[name] -= (self.lr * first_unbias)/ (np.sqrt(second_unbias) + self.epsilon)



