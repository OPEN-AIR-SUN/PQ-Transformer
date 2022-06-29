# %%
import numpy as np
from IPython import embed
import scipy.special as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt


# %%
class PDFunction:

    def __init__(self, *args) -> None:
        self.init_params = args
        self.params = [*args]

    def __call__(self, t):
        raise NotImplementedError

    def em_step(self, arr, prob):
        raise NotImplementedError


class GammaDistribution(PDFunction):

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def __call__(self, t):
        a, b = self.params
        return b**a / (sp.gamma(a)) * np.e**(-b * t) * t**(a - 1) 

    def em_step(self, arr, prob):
        target = np.log((prob * arr).sum() / prob.sum()) - (prob * np.log(arr)).sum() / prob.sum()
        coef = prob.sum() / (prob * arr).sum()
        func = lambda x: np.log(x) - sp.digamma(x) - target
        jac = lambda x: 1 / x - sp.gamma(x)
        root = opt.root(func, self.params[0], jac=jac)
        self.params[0] = root.x[0]
        self.params[1] = self.params[0] * coef


class PoissonDistribution(PDFunction):
    ...


# %%
def visualize_pdf(func: PDFunction, boundary, nstep=1000, color='green'):
    low, high = boundary
    x = np.arange(nstep) / nstep * (high - low) + low
    y = func(x)
    plt.plot(x, y, color=color, alpha=0.75)


def error_pdf(func, data_arr, steps=50000):
    y = np.histogram(data_arr, bins=steps, density=True)[0]
    x = np.arange(steps) / steps * (data_arr.max() - data_arr.min()) + data_arr.min()
    z = func(x)
    return np.abs(y - z).mean()


# %%


class FitRunner:

    def __init__(self, distribution, arr) -> None:
        self.data_arr: np.ndarray = arr
        self.weight = 0.5
        dist_cls_a, args_a = distribution[0]
        self.dist_a: PDFunction = dist_cls_a(*args_a)
        dist_cls_b, args_b = distribution[1]
        self.dist_b: PDFunction = dist_cls_b(*args_b)

    def fit(self, step=50, visualize=False, quiet=False):
        for i in range(step):
            calc = lambda x: self.weight * self.dist_a(x) + (1 - self.weight) * self.dist_b(x)
            if not quiet:
                print(f"Step #{i}")
                print(self)
                print(f"Error: {error_pdf(calc, self.data_arr)}")
            if visualize:
                self.visualize()
            pdf_a = self.dist_a(self.data_arr)
            pdf_b = self.dist_b(self.data_arr)
            pdf_sum = self.weight * pdf_a + (1 - self.weight) * pdf_b
            prob_a = self.weight * pdf_a / pdf_sum
            prob_b = (1 - self.weight) * pdf_b / pdf_sum
            self.weight = prob_a.sum() / len(prob_a)
            self.dist_a.em_step(self.data_arr, prob_a)
            self.dist_b.em_step(self.data_arr, prob_b)

    def error(self, steps=50000):
        y = np.histogram(self.data_arr, bins=steps, density=True)[0]
        x = np.arange(steps) / steps * (self.data_arr.max() - self.data_arr.min()) + self.data_arr.min()
        z = self.dist_a(x) * self.weight + self.dist_b(x) * (1 - self.weight)
        return np.abs(y - z).mean()

    def visualize(self):
        plt.hist(self.data_arr, bins=1000, alpha=0.5, density=True)
        calc = lambda x: (self.weight * self.dist_a(x) + (1 - self.weight) * self.dist_b(x))
        visualize_pdf(calc, (self.data_arr.min(), self.data_arr.max()))
        visualize_pdf(lambda x: self.weight * self.dist_a(x), (self.data_arr.min(), self.data_arr.max()), color='red')
        visualize_pdf(lambda x: (1 - self.weight) * self.dist_b(x), (self.data_arr.min(), self.data_arr.max()),
                      color='blue')
        plt.show()

    def __str__(self) -> str:
        return (f'Distribution 1 params: {self.dist_a.params}\n') + (
            f'Distribution 2 params: {self.dist_b.params}\n') + (f'Weight: {self.weight}')

def fit_gamma(arr):
    arr = np.abs(arr)
    a1, b1 = 2, 200
    a2, b2 = 50, 50
    weight = 0.5
    dist_cls = GammaDistribution
    bins = 500
    plt.hist(arr, bins=bins, alpha=0.5, density=True, stacked=True)
    dist_a, dist_b = dist_cls(a1, b1), dist_cls(a2, b2)
    visualize_pdf(lambda x: (1 - weight) * dist_b(x), (arr.min(), arr.max()), color='blue')
    visualize_pdf(lambda x: weight * dist_a(x) + (1 - weight) * dist_b(x), (arr.min(), arr.max()), color='green')
    runner = FitRunner([(dist_cls, (a1, b1)), (dist_cls, (a2, b2))], arr)
    runner.fit(quiet=True)
    mask_label = []
    for each in arr:
        if weight * dist_a(each) >= (1 - weight) * dist_b(each):
            mask_label.append(False)
        else:
            mask_label.append(True)
    return mask_label

# # %%
# # Data preparation
# arr = np.load('test.npy')
# arr = np.abs(arr)

# # %%
# # Initial params
# a1, b1 = 2, 100
# a2, b2 = 49, 100
# weight = 0.5
# dist_cls = GammaDistribution
# bins = 500

# # %%
# # Visualize data
# plt.hist(arr, bins=bins, alpha=0.5, density=True, stacked=True)
# dist_a, dist_b = dist_cls(a1, b1), dist_cls(a2, b2)
# visualize_pdf(lambda x: (1 - weight) * dist_b(x), (arr.min(), arr.max()), color='blue')
# visualize_pdf(lambda x: weight * dist_a(x) + (1 - weight) * dist_b(x), (arr.min(), arr.max()), color='green')
# plt.show()
# # %%
# # Fitting
# runner = FitRunner([(dist_cls, (a1, b1)), (dist_cls, (a2, b2))], arr)
# runner.fit(quiet=True)
# print(runner)

# # %%
