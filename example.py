import pandas as pd
import statsmodels.api as sm
from ncvgbdt import run_analysis, make_figures

df = pd.DataFrame(sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data)

base_dir = './test'

run_analysis(df,
             y_names=['mpg'],
             x_con_names=['wt'],
             x_cat_bin_names=['vs'],
             x_cat_mult_names=['gear'],
             objective='regression',
             type='CV',
             n_rep_outer_cv=1,
             n_samples_inner_cv=10,
             n_samples_rs=10,
             max_samples_shap=10,
             shap_interactions=False,
             results_dir=base_dir)

make_figures(results_dir=base_dir,
             figures_dir=base_dir)


