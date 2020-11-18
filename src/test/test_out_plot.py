import unittest


import out_plot


class TestOutPlot(unittest.TestCase):

    def test_plot_by_mu(self):
        avg_causality_dummy_by_mu = {
            10: [96.875, 98.75, 98.80, 98.875, 99],
            50: [96.875, 99, 99.10, 99.125, 99.5],
            100: [96.875, 99.25, 99, 98.60, 98.5],
            500: [96.875, 98.5, 99.125, 98.625, 99.450],
            1000: [96.875, 98.5, 99.125, 98.625, 99.375]
        }

        avg_purity_dummy_by_mu = {
            10: [62, 44, 50, 44, 42],
            50: [62, 48, 46, 42, 41],
            100: [62, 40, 44, 52, 56],
            500: [62, 98, 62, 63, 64],
            1000: [62, 70, 72, 82, 81]
        }
        out_plot.plot_for_mu(
            avg_causality_dummy_by_mu,
            avg_purity_dummy_by_mu)

    def test_plot_by_tn(self):
        avg_causality_dummy_by_tn = {
            10: [97.75, 98.85, 98.65, 99.65, 99.5],
            20: [98.550, 98.5, 98.725, 98.25, 99.15],
            30: [96.875, 98.60, 99.10, 98.625, 99.475],
            40: [97.625, 98.625, 98.875, 99.125, 98.75],
            'TNVar': [98.125, 98.95, 99.875, 99.625, 99.125]
        }

        avg_purity_dummy_by_tn = {
            10: [40, 52, 62, 55, 56],
            20: [42, 58, 78, 79, 76],
            30: [62, 70, 71, 82, 80],
            40: [48, 91, 82, 90, 88],
            'TNVar': [40, 91, 87, 85, 89]
        }
        out_plot.plot_for_tn(
            avg_causality_dummy_by_tn,
            avg_purity_dummy_by_tn)
