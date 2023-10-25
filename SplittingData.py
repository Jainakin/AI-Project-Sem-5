from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import ImportAndClean
import pandas as pd

df = ImportAndClean.cleaning()

print(df.head(3))
 
s = (df.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ',
      len(object_cols))
OH_encoder = OneHotEncoder(sparse=False)
# OH_cols = pd.DataFrame(OH_encoder.fit_transform(df[object_cols]))
# OH_cols.index = df.index
# OH_cols.columns = OH_encoder.get_feature_names()
# df_final = df.drop(object_cols, axis=1)
# df_final = pd.concat([df_final, OH_cols], axis=1)