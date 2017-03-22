from datPrep import raw,var
import pandas as pd
fullVariable= var+['target']
i=0
df=raw[fullVariable]
correlation=df.corr()
writer = pd.ExcelWriter('output\\correlation2.xlsx', engine='xlsxwriter')
correlation.to_excel(writer, 'sheet1')
writer.close()


