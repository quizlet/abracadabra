#!/usr/bin/python
# -*- coding: utf-8 -*-
from scipy import stats
from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np
from abra.utils import dict_to_object
from abra.stats import Samples
NPTS = 100
LABEL_Y_OFFSET_FACTOR = 30.

COLORS = dict_to_object(
    {
        "blue": "#4257B2",
        "light_blue": "#A1C4FD",
        "cyan": "#3CCFCF",
        "green": "#388E34",
        "light_green": "#28CC7D",
        "dark_green": "#006060",
        "yellow": "#FFCD1F",
        "salmon": "#FF725B",
        "red": "#FB3640",
        "dark_red": "#AE2024",
        "purple": "#8842C0",
        "gray": "#687174",
        "dark_gray": "#455357",
        "light_gray": "#C0CACE",
        "brown": "#665000"
    }
)

CONTROL_COLOR = COLORS.blue
VARIATION_COLOR = COLORS.green
DIFF_COLOR = COLORS.dark_gray
RESULTS_FIGSIZE = (15, 10)


class Plottable(object):
    def __init__(self, label=None, color=None):
        self.label = label
        self.color = color


class Pdf(Plottable):
    """
    Base class for plotting probability density functions.
    """
    def __init__(self, fill=True, *args, **kwargs):
        super(Pdf, self).__init__(*args, **kwargs)
        self.fill = fill

    def density(self, xs):
        """
        Evaluate
        """
        raise NotImplementedError("Implement Me")

    def xgrid(self):
        """
        Return the default x-values for plotting
        """
        raise NotImplementedError("Implement Me")

    def ppf(self, x):
        return self.dist.ppf(x)

    def cdf(self, x):
        return self.dist.cdf(x)

    def get_series(self):
        xs = self.xgrid().flatten()
        ys = self.density(xs)
        return xs, ys

    def plot(self, **plot_args):
        xs, ys, = self.get_series()
        plt.plot(xs, ys, label=self.label, color=self.color, **plot_args)
        if self.fill:
            self.plot_area(xs, ys)

    def plot_area(self, xs=None, ys=None, color=None, alpha=.25, label=None):
        xs = self.xgrid().flatten() if xs is None else xs
        ys = self.density(xs) if ys is None else ys
        color = self.color if color is None else color
        plt.fill_between(xs, ys, color=color, alpha=alpha, label=label)

    def sample(self, size):
        return self.dist.rvs(size=size)


class KdePdf(Pdf):
    """
    Estimate the shape of a PDF using a kernel density estimate.
    """
    def __init__(self, samples, *args, **kwargs):
        super(KdePdf, self).__init__(*args, **kwargs)
        self.kde = stats.gaussian_kde(samples)
        low = min(samples)
        high = max(samples)
        self._xgrid = np.linspace(low, high, NPTS + 1)

    def density(self, xs):
        return self.kde.evaluate(xs)

    def xgrid(self):
        return self._xgrid


class Pdfs(object):
    """
    Plot a sequence of Pdf instances.
    """
    def __init__(self, pdfs):
        self.pdfs = pdfs

    def plot(self):
        # labels = []
        for p in self.pdfs:
            p.plot()
        plt.legend()


class Gaussian(Pdf):
    """
    Plot a Gaussian PDF
    """
    def __init__(self, mean=0., std=1., *args, **kwargs):
        super(Gaussian, self).__init__(*args, **kwargs)
        self.mean = mean
        self.std = std
        self.dist = stats.norm(loc=mean, scale=std)

    def density(self, xs):
        return self.dist.pdf(xs)

    def xgrid(self):
        _min = self.mean - 4 * self.std,
        _max = self.mean + 4 * self.std
        return np.linspace(_min, _max, NPTS + 1)


class Pmf(Plottable):
    """
    Base class for plotting probability mass functions.
    """
    def density(self, xs):
        raise NotImplementedError("Implement Me")

    def xgrid(self):
        """
        Return the default x-values for plotting
        """
        raise NotImplementedError("Implement Me")

    def get_series(self):
        xs = self.xgrid()
        ys = self.density(xs)
        return xs, ys

    def plot(self, plot_type='step', **plot_args):
        """
        Parameters
        ---------
        plot_type: str
            The type of plot mode to use, can one of matplotlib's default plot
            types (e.g. 'bar', 'plot', 'scatter')
        """
        xs, ys, = self.get_series()
        plotfun = getattr(plt, plot_type)
        plotfun(xs, ys, label=self.label, color=self.color, **plot_args)

    def sample(self, size):
        return self.dist.rvs(size=size)


class Binomial(Pmf):
    """
    Plot a Binomial PMF
    """
    def __init__(self, n=20, p=.5, *args, **kwargs):
        super(Binomial, self).__init__(*args, **kwargs)
        self.n = n
        self.p = p
        self.dist = stats.binom(n, p)

    def density(self, xs):
        return self.dist.pmf(xs)

    def xgrid(self):
        return np.arange(0, self.n)


class Bernoulli(Pmf):
    """
    Plot a Bernoulli PDF
    """
    def __init__(self, plot_type='bar', p=0.5, *args, **kwargs):
        super(Bernoulli, self).__init__(*args, **kwargs)
        self.plot_type = plot_type
        self.p = p
        self.dist = stats.bernoulli(p)

    def density(self, xs):
        return self.dist.pmf(xs)

    def xgrid(self):
        return np.linspace(0., 1., 2)


class Poisson(Pmf):
    """
    Plot a Binomial PMF
    """
    def __init__(self, mu=1, *args, **kwargs):
        super(Poisson, self).__init__(*args, **kwargs)
        self.mu = mu
        self.dist = stats.poisson(mu)

    def density(self, xs):
        return self.dist.pmf(xs)

    def xgrid(self):
        return np.arange(0, max([1 + self.mu * 2., 11]))


def plot_interval(
    left, right, middle,
    color=None, display_text=False,
    label=None, y=0., offset=.005, fontsize=14
):
    color = color if color else 'k'
    text_y = y + offset

    if middle in (-np.inf, np.inf) and (left in (np.inf, -np.inf) or right in (np.inf, -np.inf)):
        raise ValueError('too many interval values are inf')

    _left = middle - 4 * np.abs(right) if left in (np.inf, -np.inf) else left
    _right = middle + 4 * np.abs(left) if right in (np.inf, -np.inf) else right

    plt.plot((_left, _right), (y, y), color=color, linewidth=3, label=label)
    plt.plot(middle, y, 'o', color=color, markersize=10)

    if display_text:
        label = "{}\n({}, {})".format(round(middle, 2), round(left, 2), round(right, 2))
        plt.text(middle, text_y, label, ha='center', fontsize=fontsize, color=color)


def raise_y(ax, baseline=0):
    ylims = ax.get_ylim()
    ax.set_ylim(baseline, ylims[1])
    return ax


def lower_y(ax, baseline=None):
    ylims = ax.get_ylim()

    baseline = baseline if baseline else ylims[0] - np.abs(ylims[1]) * .05

    ax.set_ylim(baseline, ylims[1])
    return ax


def visualize_gaussian_results(results, figsize=(15, 10), outfile=None, *args, **kwargs):
    """
    Visualize the results that use Gaussian approximation.
    """
    pdf_control = Gaussian(
        mean=results.control.mean,
        std=results.control.std,
        label=results.control.name,
        color=CONTROL_COLOR
    )
    pdf_variation = Gaussian(
        mean=results.variation.mean,
        std=results.variation.std,
        label=results.variation.name,
        color=VARIATION_COLOR
    )
    pdfs = Pdfs([pdf_control, pdf_variation])

    mean_diff = results.variation.mean - results.control.mean
    std_diff = ((results.control.var / results.control.nobs) + \
                (results.variation.var / results.control.nobs)) ** .5
    pdf_diff = Gaussian(mean_diff, std_diff, label='Difference', color=DIFF_COLOR)

    fig, axs = plt.subplots(3, 1, figsize=figsize)
    plt.sca(axs[0])
    pdfs.plot()
    raise_y(axs[0])
    plt.gca().get_yaxis().set_ticks([])
    plt.title("Sample Comparison")
    x_min, x_max = plt.xlim()

    plt.sca(axs[1])
    y_min, y_max = plt.ylim()
    y_dist = (y_max - y_min) / LABEL_Y_OFFSET_FACTOR
    plot_interval(
        *results.control.std_err(),
        middle=results.control.mean,
        y=y_dist,
        offset=-.015,
        color=CONTROL_COLOR,
        display_text=True,
        label=results.control.name
    )
    plot_interval(
        *results.variation.std_err(),
        middle=results.variation.mean,
        y=-y_dist,
        offset=0.005,
        color=VARIATION_COLOR,
        display_text=True,
        label=results.variation.name
    )
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.gca().get_yaxis().set_ticks([])
    plt.title("Mean +/- Standard Error")

    # plot differences distribution
    plt.sca(axs[2])
    plt.axvline(0., color=DIFF_COLOR, linestyle='--', linewidth=1.5)

    if results.inference_procedure.hypothesis == 'larger':
        left_bound = results.ci[0][0]
        right_bound = np.inf
    elif results.inference_procedure.hypothesis == 'smaller':
        right_bound = results.ci[0][1]
        left_bound = np.inf
    else:
        left_bound = results.ci[0][0]
        right_bound = results.ci[0][1]

    plot_interval(
        left_bound,
        right_bound,
        mean_diff,
        color=DIFF_COLOR,
        display_text=True
    )
    plt.gca().get_yaxis().set_ticks([])
    plt.title(results.comparison_type)
    if outfile:
        plt.savefig(
            outfile,
            bbox_inches='tight',
            dpi=300
        )


def visualize_binomial_results(results, figsize=(15, 10), outfile=None, *args, **kwargs):
    """
    Visualize the results that use Gaussian approximation.
    """
    tol = 1e-4

    pmf_control = Binomial(
        p=results.control.mean,
        n=results.control.nobs,
        label=results.control.name,
        color=CONTROL_COLOR
    )

    pmf_variation = Binomial(
        p=results.variation.mean,
        n=results.variation.nobs,
        label=results.variation.name,
        color=VARIATION_COLOR
    )

    xy_control = zip(pmf_control.xgrid(), pmf_control.density(pmf_control.xgrid()))
    xy_variation = zip(pmf_variation.xgrid(), pmf_variation.density(pmf_variation.xgrid()))

    valid_xy_control = sorted([x for x in xy_control if x[1] >= tol], key=lambda x: x[0])
    valid_xy_variation = sorted([x for x in xy_variation if x[1] >= tol], key=lambda x: x[0])

    x_min = int(min(valid_xy_control[0][0], valid_xy_variation[0][0]))
    x_max = int(max(valid_xy_control[-1][0], valid_xy_variation[-1][0]))

    mean_diff = results.variation.mean - results.control.mean
    std_diff = (results.control.var / results.control.nobs + \
                results.variation.var / results.control.nobs) ** .5
    pdf_diff = Gaussian(mean_diff, std_diff, label='Difference', color=DIFF_COLOR)

    fig, axs = plt.subplots(3, 1, figsize=figsize)
    plt.sca(axs[0])

    # make plotting more scalable
    if pmf_control.n > 1000 or pmf_variation.n > 1000:
        plot_type = 'step'
    else:
        plot_type = 'bar'

    pmf_control.plot(plot_type=plot_type, alpha=.5)
    pmf_variation.plot(plot_type=plot_type, alpha=.5)
    raise_y(axs[0])
    plt.xlim(x_min, x_max)
    # plt.gca().get_xaxis().set_ticks([])
    # plt.gca().get_yaxis().set_ticks([])
    plt.legend()
    plt.title("Sample Comparison")

    plt.sca(axs[1])
    y_min, y_max = plt.ylim()
    y_dist = (y_max - y_min) / LABEL_Y_OFFSET_FACTOR
    plot_interval(
        *results.control.std_err(),
        middle=results.control.mean,
        y=y_dist,
        offset=-0.015,
        color=CONTROL_COLOR,
        display_text=True,
        label=results.control.name
    )
    plot_interval(
        *results.variation.std_err(),
        middle=results.variation.mean,
        y=-y_dist,
        offset=0.005,
        color=VARIATION_COLOR,
        display_text=True,
        label=results.variation.name
    )

    plt.legend()
    plt.gca().get_yaxis().set_ticks([])
    plt.title("Proportions +/- Standard Error")

    # Differences plot
    plt.sca(axs[2])
    plt.axvline(0., color=DIFF_COLOR, linestyle='--', linewidth=1.5)

    # xs = pdf_diff.xgrid()
    if results.inference_procedure.hypothesis == 'larger':
        left_bound = results.ci[0][0]
        right_bound = np.inf
    elif results.inference_procedure.hypothesis == 'smaller':
        right_bound = results.ci[0][1]
        left_bound = np.inf
    else:
        left_bound = results.ci[0][0]
        right_bound = results.ci[0][1]

    plot_interval(
        left_bound,
        right_bound,
        mean_diff,
        color=DIFF_COLOR,
        display_text=True
    )
    plt.gca().get_yaxis().set_ticks([])
    plt.title(results.comparison_type)
    if outfile:
        plt.savefig(
            outfile,
            bbox_inches='tight',
            dpi=300
        )


def visualize_rates_results(results, figsize=(15, 10), outfile=None, *args, **kwargs):
    fig, axs = plt.subplots(3, 1, figsize=figsize)

    # Sample Comparison plot
    plt.sca(axs[0])
    control_pmf = Poisson(
        results.control.mean,
        color=CONTROL_COLOR,
        label=results.control.name
    )

    variation_pmf = Poisson(
        results.variation.mean,
        color=VARIATION_COLOR,
        label=results.variation.name
    )
    control_pmf.plot(plot_type='bar', alpha=.5)
    variation_pmf.plot(plot_type='bar', alpha=.5)
    plt.legend()
    plt.title("Sample Comparison")

    # Rates +/- standard error plot
    plt.sca(axs[1])
    y_min, y_max = plt.ylim()
    y_dist = (y_max - y_min) / LABEL_Y_OFFSET_FACTOR
    plot_interval(
        *results.control.std_err(),
        middle=results.control.mean,
        y=y_dist,
        offset=-0.015,
        color=CONTROL_COLOR,
        display_text=True,
        label=results.control.name
    )
    plot_interval(
        *results.variation.std_err(),
        middle=results.variation.mean,
        y=-y_dist,
        offset=0.005,
        color=VARIATION_COLOR,
        display_text=True,
        label=results.variation.name
    )
    plt.legend()
    plt.gca().get_yaxis().set_ticks([])
    plt.title("Rates +/- Standard Error")

    # Differences plot
    plt.sca(axs[2])

    plot_interval(
        *results.ci[0],
        middle=results.delta,
        color=DIFF_COLOR,
        display_text=True
    )
    plt.axvline(1., color=DIFF_COLOR, linestyle='--', linewidth=1.5)
    plt.gca().get_yaxis().set_ticks([])
    plt.title(results.comparison_type)
    if outfile:
        plt.savefig(
            outfile,
            bbox_inches='tight',
            dpi=300
        )


def visualize_bootstrap_results(results, figsize=(15, 10), outfile=None, plot_type='bar', *args, **kwargs):
    fig, axs = plt.subplots(3, 1, figsize=figsize)

    # Sample Comparison plot
    plt.sca(axs[0])
    
    if plot_type == 'bar':
        bins = 50 if results.control.nobs >= 100 or results.variation.nobs >= 100 else 20
        results.control.hist(bins=bins, color=CONTROL_COLOR, alpha=.5, label=results.control.name)
        results.variation.hist(bins=bins, color=VARIATION_COLOR, alpha=.5, label=results.variation.name)
    else:
        control_pmf = KdePdf(
            samples=results.control.data,
            color=CONTROL_COLOR,
            label=results.control.name
        )

        variation_pmf = KdePdf(
            samples=results.variation.data,
            color=VARIATION_COLOR,
            label=results.variation.name
        )
        control_pmf.plot(alpha=.5)
        variation_pmf.plot(alpha=.5)

    plt.legend()
    plt.title("Sample Comparison")

    # Bootstrapped statistic +/- HDI
    plt.sca(axs[1])
    y_min, y_max = plt.ylim()
    y_dist = (y_max - y_min) / LABEL_Y_OFFSET_FACTOR
    plot_interval(
        *results.aux['control'].hdi(),
        middle=results.aux['control'].mean,
        y=y_dist,
        offset=-0.015,
        color=CONTROL_COLOR,
        display_text=True,
        label=results.control.name
    )
    plot_interval(
        *results.aux['variation'].hdi(),
        middle=results.aux['variation'].mean,
        y=-y_dist,
        offset=0.005,
        color=VARIATION_COLOR,
        display_text=True,
        label=results.variation.name
    )
    plt.legend()
    plt.gca().get_yaxis().set_ticks([])
    plt.title(f"Bootstrap({results.test_statistic}) +/- 95% HDI")

    # Differences plot
    plt.sca(axs[2])

    plot_interval(
        *results.ci[0],
        middle=results.delta,
        color=DIFF_COLOR,
        display_text=True
    )
    plt.axvline(0., color=DIFF_COLOR, linestyle='--', linewidth=1.5)
    plt.gca().get_yaxis().set_ticks([])
    plt.title(f"{results.comparison_type}({results.test_statistic})")
    if outfile:
        plt.savefig(
            outfile,
            bbox_inches='tight',
            dpi=300
        )

def visualize_bayesian_results(results, figsize=RESULTS_FIGSIZE, outfile=None, *args, **kwargs):
    fig, axs = plt.subplots(3, 1, figsize=figsize)

    def get_central_tendency_params(results):
        if 'p_control' in results.traces.variables:
            return 'p_control', 'p_variation', '$p$ (proportion)'
        elif 'mu_control' in results.traces.variables:
            return 'mu_control', 'mu_variation', '$\\mu$ (mean)'
        elif 'lambda_control' in results.traces.variables:
            return 'lambda_control', 'lambda_variation', '$\\lambda$ (rate)'
    HDI = 0.95
    HDI_PRCT = round(HDI * 100)
    plt.sca(axs[0])
    ctps = get_central_tendency_params(results)
    results.traces.plot(
        ctps[0],
        label=results.control.name,
        color=COLORS.blue,
        alpha=.4
    )
    results.traces.plot(
        ctps[1],
        label=results.variation.name,
        color=COLORS.green,
        alpha=.4,
        title='Comparison of {}'.format(ctps[2])
    )
    plt.legend()
    lower_y(axs[0])
    x_min, x_max = plt.xlim()

    plt.sca(axs[1])
    y_min, y_max = plt.ylim()
    y_dist = (y_max - y_min) / LABEL_Y_OFFSET_FACTOR
    results.traces.plot(
        ctps[0],
        label=results.control.name,
        color=COLORS.blue,
        hdi=HDI,
        include_hist=False,
        y=y_dist,
        offset=-0.015
    )

    results.traces.plot(
        ctps[1],
        label=results.variation.name,
        color=COLORS.green,
        hdi=HDI,
        include_hist=False,
        y=-y_dist,
        offset=0.005
    )
    lower_y(axs[1])
    plt.xlim([x_min, x_max])
    plt.title(f"{ctps[2]} +/- {HDI_PRCT}% HDI")

    plt.sca(axs[2])
    results.traces.plot(
        'delta',
        hdi=1 - results.alpha,
        ref_val=0.0,
        color=COLORS.dark_gray,
        title='Differences in {}'.format(ctps[2])
    )
    lower_y(axs[2])

    if outfile:
        plt.savefig(
            outfile,
            bbox_inches='tight',
            dpi=300
        )

class Traces(object):
    """
    Container class for analyzing the results of Bayesian inference procedure.

    Parameters
    ----------
    traces: dict
        Key-value pairs of parameters:samples, extracted from a Bayesian inference
        procedure.
    """

    def __init__(self, traces):
        self.variables = []
        for k, v in list(traces.items()):
            if k != "lp__":
                self.variables.append(k)
                setattr(self, k, Samples(v))
        self.summarize()

    def summarize(self):
        prct = [2.5, 25, 50, 75, 97.5]
        values = []
        columns = []
        for v in self.variables:
            trace = getattr(self, v)
            _mean = trace.mean
            _hdi = trace.hdi()
            _std = trace.std
            _percentiles = trace.percentiles(prct)
            values.append(np.r_[_mean, _hdi, _std, _percentiles])
        columns = ['mean', 'hdi_lower', 'hdi_upper', 'std'] + ["{}%".format(p) for p in prct]
        self._summary = DataFrame(values, columns=columns, index=self.variables)

    @property
    def summary(self):
        return self._summary

    def plot(
        self, variable, label=None,
        color=None, ref_val=None, alpha=.25,
        bins=None, title=None,
        hdi=None, outfile=None,
        include_hist=True,
        offset=5,
        y=0.
    ):
        """
        Plot the histogram of a variable trace

        Parameters
        ----------
        variable : str
            The name of one of self.variables to plot
        label : str
            Alternative label for the legend
        ref_val : float
            A reference value location at which to draw a vertical line
        alpha : float in [0 1)
            The transparency of the histogram. Ignored if `include_hist=False`
        bins : int
            The number of histogram bins. Ignored if `include_hist=False`
        title : str
            The title of the plot
        hdi : float in [0, 1]
            The amount of probability mass within the Highest Density Interval
            to display on the histogram.
        y : float
            The y offset for interval plots. Ignored if `hdi is None`
        offset : float
            The text offset for interval plots. Ignored if `hdi is None`
        outfile : str
            The name of an output file to save the figure to.
        """
        from matplotlib import pyplot as plt  # lazy import
        from abra.vis import plot_interval

        if (include_hist is False) and (hdi is None):
            raise ValueError('include_hist must be True if hdi is None')

        if variable not in self.variables:
            raise ValueError('Variable `{}` not available'.format(variable))

        label = label if label else variable
        trace = getattr(self, variable)
        
        if include_hist:
            if bins is None:
                bins = int(len(trace.data) / 50.)
            trace.hist(
                color=color,
                alpha=alpha,
                bins=bins,
                ref_val=ref_val,
                label=label
            )
        
        if hdi is not None:  # highest density interval
            median = round(trace.percentiles(50), 3)
            _hdi = [round(h, 3) for h in trace.hdi(1 - hdi)]
            plot_interval(
                *_hdi,
                middle=median,
                display_text=True,
                offset=offset,
                y=y,
                color=color
            )

        if title is None:
            if ref_val is not None:
                gt = round(100 * trace.prob_greater_than(ref_val))
                title = " {}% < {} = {} < {}%".format(100 - gt, variable, ref_val, gt)
            else:
                title = ''
        plt.title(title, fontsize=14)

        if outfile:
            plt.savefig(
                outfile,
                bbox_inches='tight',
                dpi=300
            )
