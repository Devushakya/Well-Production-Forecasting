"""Adapted production forecasting workflow; train-only decline fitting."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import json

ROOT = Path(__file__).resolve().parents[1]

def arps(t, qi, di, b):
    return qi / (1 + b * di * t)**(1 / b)

def clean(raw, well):
    d = raw.loc[raw.NPD_WELL_BORE_NAME.eq(well), ['DATEPRD','BORE_OIL_VOL','ON_STREAM_HRS']].copy()
    d['DATEPRD'] = pd.to_datetime(d.DATEPRD, errors='coerce')
    for c in ['BORE_OIL_VOL','ON_STREAM_HRS']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna().sort_values('DATEPRD')
    d = d[d.ON_STREAM_HRS.between(2,24) & (d.BORE_OIL_VOL > 0)].copy()
    if d.DATEPRD.duplicated().any():
        raise ValueError('Duplicate daily records: inspect workbook before training.')
    d['rate'] = 24*d.BORE_OIL_VOL/d.ON_STREAM_HRS
    return d

def run(path, well, out):
    raw = pd.read_excel(path, sheet_name='Daily Production Data')
    d = clean(raw, well)
    if len(d) < 100: raise ValueError('Insufficient flowing observations for this well.')
    # Hold out final 20% before choosing decline start.
    cutoff = d.DATEPRD.iloc[int(len(d)*.8)]
    historical = d[d.DATEPRD < cutoff].copy()
    rolling = historical.rate.rolling(30, min_periods=30).mean()
    start = historical.loc[rolling >= .9*rolling.max(), 'DATEPRD'].max()
    d = d[d.DATEPRD >= start].copy()
    d['days'] = (d.DATEPRD-start).dt.days
    # Lags are previous flowing observations, not necessarily calendar days.
    features = ['days']
    for k in (1,3,7,14):
        name = f'lag_{k}'; d[name] = d.rate.shift(k); features.append(name)
    d = d.dropna(subset=features)
    train = d[d.DATEPRD < cutoff]; test = d[d.DATEPRD >= cutoff]
    if len(train)<30 or len(test)<2: raise ValueError('Insufficient decline data after chronological split.')
    params,_ = curve_fit(arps, train.days, train.rate, p0=[train.rate.iloc[0],.001,.5], bounds=([0,1e-8,.001],[np.inf,1,1.5]), maxfev=20000)
    models = {'Random Forest':RandomForestRegressor(n_estimators=200,max_depth=10,random_state=42,n_jobs=2), 'XGBoost':XGBRegressor(n_estimators=200,max_depth=5,learning_rate=.05,random_state=42,n_jobs=2)}
    pred = {'Arps':arps(test.days.to_numpy(),*params), 'Persistence':test.lag_1.to_numpy()}
    out.mkdir(parents=True,exist_ok=True)
    for name,model in models.items():
        model.fit(train[features],train.rate)
        pred[name] = model.predict(test[features])
        joblib.dump(model,out/(name.lower().replace(' ','_')+'.joblib'))
    metrics = pd.DataFrame([{'model':n,'r2':r2_score(test.rate,p),'rmse':np.sqrt(mean_squared_error(test.rate,p)),'mae':mean_absolute_error(test.rate,p)} for n,p in pred.items()])
    metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame({'date':test.DATEPRD,'actual':test.rate,**pred}).to_csv(out/'predictions.csv',index=False)
    (out/'run_info.json').write_text(json.dumps({'well':well,'cutoff':str(cutoff),'decline_start':str(start),'arps_parameters':list(map(float,params)),'features':features,'train_rows':len(train),'test_rows':len(test)},indent=2))
    fig,ax=plt.subplots(figsize=(12,5)); ax.scatter(test.DATEPRD,test.rate,s=8,label='Observed',alpha=.55)
    for n,p in pred.items(): ax.plot(test.DATEPRD,p,label=n,linewidth=1)
    ax.set(xlabel='Date',ylabel='Normalized oil rate (Sm³/day)',title=f'{well}: held-out sequential predictions'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out/'forecast.png',dpi=150); plt.close(fig)
    print(metrics.to_string(index=False)); print('Results:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=ROOT/'data/raw/Volve production data.xlsx'); p.add_argument('--well',default='15/9-F-12'); p.add_argument('--output',type=Path,default=ROOT/'outputs'); a=p.parse_args(); run(a.data,a.well,a.output)
