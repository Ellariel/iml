import pandas as pd
import statsmodels.api as sm
from ncvgbdt import run_analysis, make_figures

df = pd.DataFrame(sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data)

base_dir = './test'

run_analysis(df,
             Y_NAMES = ['mpg'],
             X_CON_NAMES = ['wt'],
             X_CAT_BIN_NAMES = ['vs'],
             X_CAT_MULT_NAMES = ['gear'],
             OBJECTIVE = 'regression',
             TYPE = 'CV',
             N_REP_OUTER_CV = 1,
             N_SAMPLES_INNER_CV = 10,
             N_SAMPLES_RS = 10,
             MAX_SAMPLES_SHAP = 10,
             SHAP_INTERACTIONS = False,
             RESULTS_DIR = base_dir)

make_figures(RESULTS_DIR = base_dir,
             FIGURES_DIR = base_dir)


