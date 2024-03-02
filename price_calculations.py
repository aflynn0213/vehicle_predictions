import pandas as pd    

def calc_exp_prices(feats_w_trims,trims,model):
    temp_feats = feats_w_trims.copy()
    length = len(temp_feats.columns)
    
    trims_tmp = pd.get_dummies(trims)

    for col in range(len(trims_tmp.columns)):
        name_ = trims_tmp.columns[col]
        tmp = pd.Series(0,index=feats_w_trims.index,name=name_)
        temp_feats = pd.concat([temp_feats,tmp], axis=1)

    exp_prices = dict()
    for i in range(len(trims_tmp.columns)) :
        tmp = temp_feats.copy()
        tmp.iloc[:,length+i] = 1
        exp_prices[trims_tmp.columns[i]] = (model.prediction(tmp))
    exp_prices = pd.DataFrame(exp_prices,index=feats_w_trims.index)

    return exp_prices

def calc_test_prices(test_data,clf,reg,exp_price):
    #DEFINE SERIES USED FOR POPULATING PREDICTIONS
    test_preds_price = pd.Series(index=test_data.index)

    #WITH POSTERIOR PROBABILITY CALCULATED BY EXP PRICES WITH ONE HOT ENCODED MASKING EARLIER
    for i in test_data.index:
        test_preds_price.loc[i] = (exp_price.loc[i,:].values*clf.preds_proba.loc[i,:].values).sum()

    return test_preds_price