import pandas as pd
lgb = pd.read_csv('submission_lgb.csv')
catboost = pd.read_csv('best_ctb.csv')
neuro = pd.read_csv('preds14.csv')
columns = list(catboost.columns)
columns.remove('id')
lgb[columns] = 0.45 * catboost[columns] + 0.45 * neuro[columns] + 0.1 * lgb[columns]
lgb.to_csv('all_blend.csv', index=False)
