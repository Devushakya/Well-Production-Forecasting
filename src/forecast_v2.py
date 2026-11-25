"""Sequential production prediction; extension of the existing forecast.py workflow."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecast import clean, arps, ROOT


def fit_decline(d, window):
    s = d if window == 0 else d.tail(window)
    origin = s.days.iloc[0]
    t = s.days.to_numpy() - origin
    p, _ = curve_fit(arps, t, s.rate, p0=[s.rate.iloc[0], .001, .5],
                     bounds=([0,1e-8,.001],[np.inf,1,1.5]), maxfev=20000)
    return origin, p


def run(path, well, out):
    d = clean(pd.read_excel(path, sheet_name='Daily Production Data'), well)
    if len(d)<100: raise ValueError('Insufficient data')
    cutoff = d.DATEPRD.iloc[int(len(d)*.8)]
    h = d[d.DATEPRD < cutoff]
    rolling = h.rate.rolling(30, min_periods=30).mean()
    start = h.loc[rolling >= .9*rolling.max(), 'DATEPRD'].max()
    d = d[d.DATEPRD >= start].copy()
    d['days'] = (d.DATEPRD-start).dt.days
    for k in (1,3,7,14): d[f'lag_{k}'] = d.rate.shift(k)
    past = d.rate.shift(1)
    for k in (3,7,14):
        d[f'mean_{k}'] = past.rolling(k).mean()
    d['trend_3'] = (d.lag_1-d.lag_3)/2
    d['trend_7'] = (d.lag_1-d.lag_7)/6
    d['gap_days'] = d.DATEPRD.diff().dt.days
    features = ['lag_1','lag_3','lag_7','lag_14','mean_3','mean_7','mean_14','trend_3','trend_7','gap_days']
    d = d.dropna(subset=features).copy()
    train = d[d.DATEPRD < cutoff].copy(); test = d[d.DATEPRD >= cutoff].copy()
    if len(train)<100 or len(test)<2: raise ValueError('Insufficient train/test rows')
    a = train.iloc[:int(len(train)*.8)]; v = train.iloc[int(len(train)*.8):]
    # Select decline window using only chronological training validation.
    scores = {}
    for w in (0,90,180,365,730):
        if w>len(a): continue
        try:
            o,p = fit_decline(a,w)
            scores[w] = float(mean_absolute_error(v.rate, arps(v.days.to_numpy()-o,*p)))
        except (RuntimeError, ValueError): continue
    if not scores: raise ValueError('All decline fits failed')
    window = min(scores,key=scores.get)
    o,p = fit_decline(train,window)
    full_o,full_p = fit_decline(train,0)
    pred = {'Arps original':arps(test.days.to_numpy()-full_o,*full_p),
            'Arps recent window':arps(test.days.to_numpy()-o,*p),
            'Persistence':test.lag_1.to_numpy()}
    # Model changes from the last observed rate rather than absolute rate.
    models = {'RF delta':RandomForestRegressor(n_estimators=200,max_depth=6,min_samples_leaf=12,random_state=42,n_jobs=2),
              'XGB delta':XGBRegressor(n_estimators=150,max_depth=3,learning_rate=.03,random_state=42,n_jobs=2,objective='reg:squarederror')}
    out.mkdir(parents=True,exist_ok=True)
    shrinkage = {}
    for name,m in models.items():
        m.fit(a[features],a.rate-a.lag_1)
        correction=m.predict(v[features]); base=v.lag_1.to_numpy()
        # Zero is a valid choice: do not force a worse ML correction.
        weight=min([0,.25,.5,.75,1],key=lambda w: mean_absolute_error(v.rate,np.maximum(0,base+w*correction)))
        shrinkage[name]=weight
        m.fit(train[features],train.rate-train.lag_1)
        pred[name]=np.maximum(0,test.lag_1.to_numpy()+weight*m.predict(test[features]))
        joblib.dump({'model':m,'features':features,'correction_weight':weight},out/(name.replace(' ','_').lower()+'.joblib'))
    metrics=pd.DataFrame([dict(model=n,r2=r2_score(test.rate,y),rmse=np.sqrt(mean_squared_error(test.rate,y)),mae=mean_absolute_error(test.rate,y)) for n,y in pred.items()])
    metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame({'date':test.DATEPRD,'actual':test.rate,**pred}).to_csv(out/'predictions.csv',index=False)
    (out/'run_info.json').write_text(json.dumps(dict(well=well,cutoff=str(cutoff),decline_start=str(start),train_rows=len(train),test_rows=len(test),features=features,decline_window_observations=window,validation_mae_by_window=scores,arps_parameters=list(map(float,p)),arps_origin_days=float(o),correction_weights=shrinkage),indent=2))
    fig,ax=plt.subplots(figsize=(12,5)); ax.plot(train.DATEPRD,train.rate,label='Training history',alpha=.6); ax.plot(test.DATEPRD,test.rate,label='Test history'); ax.axvline(cutoff,ls='--',color='gray',label='Cutoff'); ax.set(ylabel='Normalized oil rate (Sm³/day)',title=well+' production history'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out/'history.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); ax.scatter(test.DATEPRD,test.rate,s=7,label='Observed',alpha=.5)
    for n,y in pred.items(): ax.plot(test.DATEPRD,y,label=n,lw=1)
    ax.set(ylabel='Normalized oil rate (Sm³/day)',title=well+' | held-out sequential predictions'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out/'forecast.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4)); ax.barh(metrics.model,metrics.mae); ax.set(xlabel='MAE (Sm³/day; lower is better)',title='Held-out model comparison'); fig.tight_layout(); fig.savefig(out/'model_comparison.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5));
    for n,y in pred.items(): ax.plot(test.DATEPRD,test.rate.to_numpy()-y,label=n,lw=.8,alpha=.75)
    ax.axhline(0,color='gray'); ax.set(ylabel='Observed minus predicted (Sm³/day)',title='Held-out residuals'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out/'residuals.png',dpi=150); plt.close(fig)
    print(metrics.to_string(index=False)); print('Validation-selected decline window:',window,'(0 = all training)'); print('ML correction weights:',shrinkage); print('Results:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=ROOT/'data/raw/Volve production data.xlsx'); p.add_argument('--well',default='15/9-F-12'); p.add_argument('--output',type=Path,default=ROOT/'outputs_v2'); a=p.parse_args(); run(a.data,a.well,a.output)
