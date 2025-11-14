import pandas as pd
import os
import joblib
import json
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import argparse

def fe(df):
    # basic one-hot for channel
    df2 = df.copy()
    df2 = pd.get_dummies(df2, columns=['channel','audience','product'], drop_first=True)
    return df2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/synthetic_ads.csv')
    parser.add_argument('--model_out', default='models/ctr_xgb.json')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    df = pd.read_csv(args.data)
    df = df.dropna(subset=['ctr'])
    df2 = fe(df)
    y = df2['ctr']
    X = df2.drop(['ad_id','date','clicks','conversions','spend','ctr','cvr'], axis=1, errors='ignore')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {"objective":"reg:squarederror", "eval_metric":"mae", "learning_rate":0.05}
    model = xgb.train(params, dtrain, num_boost_round=500, evals=[(dval,'val')], early_stopping_rounds=25)
    model.save_model(args.model_out)
    # save feature list
    meta = {'features': list(X.columns)}
    with open(os.path.join(os.path.dirname(args.model_out),'feature_metadata.json'), 'w') as f:
        json.dump(meta, f)
    preds = model.predict(dval)
    print('VAL MAE', mean_absolute_error(y_val, preds))
    joblib.dump({'mae': mean_absolute_error(y_val, preds)}, os.path.join(os.path.dirname(args.model_out),'train_metrics.joblib'))

if __name__=='__main__':
    main()
