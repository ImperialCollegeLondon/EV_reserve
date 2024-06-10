def trainpick(df_x):
    df_train = df_x.iloc[:12240,:]
    return df_train

def testpick(df_x):
    df_test = df_x.iloc[12240:,:]
    return df_test