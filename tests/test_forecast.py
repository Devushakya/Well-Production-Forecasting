import unittest
import numpy as np
import pandas as pd
from src.forecast import arps, clean
class Checks(unittest.TestCase):
    def test_decline(self):
        self.assertEqual(arps(0,1000,.01,.5),1000)
        self.assertTrue(np.all(np.diff(arps(np.arange(10),1000,.01,.5))<0))
    def test_normalization(self):
        df=pd.DataFrame({'NPD_WELL_BORE_NAME':['A','A'],'DATEPRD':['2020-01-01','2020-01-02'],'BORE_OIL_VOL':[100,0],'ON_STREAM_HRS':[12,0]})
        self.assertEqual(clean(df,'A').rate.tolist(),[200])
if __name__=='__main__': unittest.main()
