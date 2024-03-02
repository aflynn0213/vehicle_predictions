import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (make_scorer,
                             classification_report,                             
                             mean_squared_error, 
                             r2_score, 
                             accuracy_score, 
                             roc_auc_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix)

from xgboost import XGBRegressor, XGBClassifier
import price_calculations as pr
from sklearn.metrics import explained_variance_score

class XGB_Classifier:

    def __init__(self,feats,trims,stats,encoder):
        self.model = []
        self.stats_model = []
        self.preds_proba = []
        self.preds = []
        self.encoder = encoder

        if stats:
            self.stats_model = self.train_model(feats,trims,True)
        else:
            self.model = self.train_model(feats,trims,False)
    
    def train_model(self,feats,trims,stats):
        if (stats):
            #Split for stats on training data
            x_train, x_test, y_train, y_test = train_test_split(feats, trims, test_size=0.20, random_state=42)
        else:
            x_train = feats
            y_train = trims
        
        pars_clf = {
            'n_estimators': [200],
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1]
        }
        
        xgb_clf = XGBClassifier()
        
        scoring = {
            'precision': make_scorer(precision_score, average='weighted'),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr'),
            'accuracy': make_scorer(accuracy_score),
            'recall': make_scorer(recall_score,average='weighted')
        }
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        
        xgb_cv = GridSearchCV(xgb_clf,param_grid=pars_clf,scoring=scoring,cv=skf, refit='roc_auc')
        xgb_cv.fit(x_train,y_train)
        
        # Best parameters and best score
        print("Best Parameters: ", xgb_cv.best_params_)
        print("Best Score: ", xgb_cv.best_score_)
        
        # Grab CV best estimator
        best_xgb_clf = xgb_cv.best_estimator_

        
        if (stats):
            preds = best_xgb_clf.predict(x_test)
            preds_proba = best_xgb_clf.predict_proba(x_test)
        
            print(preds)
            print(preds_proba)
        
            # Calculates ROC-AUC score
            roc_auc_scores = []
            for i in range(len(best_xgb_clf.classes_)):
                roc_auc_i = roc_auc_score((y_test == best_xgb_clf.classes_[i]).astype(int), preds_proba[:, i])
                roc_auc_scores.append(roc_auc_i)
            
            # Average ROC-AUC score across all classes
            print(np.mean(roc_auc_scores))

            for metric in scoring:
                score_key = f"mean_test_{metric}" 
                score = xgb_cv.cv_results_[score_key]
                print(f"METRIC '{metric}': SCORES '{score}'")
        
            # Precisions
            precision_micro = precision_score(y_test, preds, average='micro')
            precision_macro = precision_score(y_test, preds, average='macro')
            precision_weighted = precision_score(y_test, preds, average='weighted')
            precision_none = precision_score(y_test, preds, average=None)
            
            print("Precision Micro:", precision_micro)
            print("Precision Macro:", precision_macro)
            print("Precision Weighted:", precision_weighted)
            print("Precision for each class:", precision_none)
            
            # Recall
            recall_micro = recall_score(y_test, preds, average='micro')
            recall_macro = recall_score(y_test, preds, average='macro')
            recall_weighted = recall_score(y_test, preds, average='weighted')
            recall_none = recall_score(y_test, preds, average=None)
            
            print("Recall Micro:", recall_micro)
            print("Recall Macro:", recall_macro)
            print("Recall Weighted:", recall_weighted)
            print("Recall for each class:", recall_none)
            
            # F1 Score
            f1_micro = f1_score(y_test, preds, average='micro')
            f1_macro = f1_score(y_test, preds, average='macro')
            f1_weighted = f1_score(y_test, preds, average='weighted')
            f1_none = f1_score(y_test, preds, average=None)
            
            print("F1 Score Micro:", f1_micro)
            print("F1 Score Macro:", f1_macro)
            print("F1 Score Weighted:", f1_weighted)
            print("F1 Score for each class:", f1_none)
            
            print(np.unique(y_test))
            print(np.unique(preds_proba))
            
            # Extend predicted probabilities to include missing classes
            # Compute ROC AUC score for micro, macro, and weighted averaging
            roc_auc_ovo = roc_auc_score(y_test, preds_proba, multi_class='ovo')
            roc_auc_ovr = roc_auc_score(y_test, preds_proba, multi_class='ovr')
            
            print("*********ROC AUC OVO*******:", roc_auc_ovo)
            print("*********ROC AUC OVR*******:", roc_auc_ovr)
            
            # Confusion Matrix
            conf_matrix = confusion_matrix(y_test, preds)
            print("Confusion Matrix:")
            print(conf_matrix)

        return best_xgb_clf
    
    def prediction(self,feats_data):
        if (self.model):
            self.preds = self.model.predict(feats_data.values)
            self.preds = pd.Series( self.encoder.inverse_transform(self.preds),
                                    index=feats_data.index,
                                    name="vehicle_trim"    )
            
            #GET CLASS PROBABILITIES USING BEST JEEP TRIM CLASSIFIER FROM TRAINING
            self.preds_proba = self.model.predict_proba(feats_data.values)
            self.preds_proba = pd.DataFrame(self.preds_proba,index=feats_data.index) 
 
class XGB_Regressor:

    def __init__(self,feats,target,trims,encoder,stats,dummies):
        self.model = []
        self.stats_model = []
        self.preds = []
        self.target = target
        if stats:
            self.stats_model = self.train_model(feats,target,trims,encoder,True,dummies)
        else:
            self.model = self.train_model(feats,target,trims,encoder,False,dummies)

    def train_model(self,feats,target,trims,encoder,stats,dummies):
        if (stats):
            x_train, x_test, y_train, y_test = train_test_split(feats, target, test_size=0.20, random_state=42,stratify=trims)
            if (dummies):
                subset_indices = x_train.index.intersection(feats.index)
                tmp_trims = pd.get_dummies(encoder.inverse_transform(trims.loc[subset_indices]))
                tmp_trims.set_index(subset_indices, inplace=True)
                print("DUMMIES: ",tmp_trims)
                x_train = pd.concat([x_train , tmp_trims] ,  axis = 1)
                print("SECOND : ",feats)
        else:
            x_train = feats
            y_train = target
            if (dummies):
                tmp_trims = pd.get_dummies(encoder.inverse_transform(trims))
                tmp_trims.set_index(feats.index,inplace=True)
                print("DUMMIES: ",tmp_trims)
                x_train = pd.concat([x_train , tmp_trims] ,  axis = 1)
                print("SECOND : ",feats)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        print(x_train)
        xgb = XGBRegressor()
        # Grid search using cross-validation
        cv = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=7, scoring='r2')
        cv.fit(x_train,y_train)
        
        # Best parameters and best score
        print("Best Parameters: ", cv.best_params_)
        print("Best Score R2: ", cv.best_score_)
        
        best_xgb_reg = cv.best_estimator_
        if (stats):
            subset_indices_test = x_test.index.intersection(feats.index)
            self.model = best_xgb_reg
            dummy_trims = pd.get_dummies(trims.loc[subset_indices_test])
            exp_prices = pr.calc_exp_prices(x_test,encoder.inverse_transform(dummy_trims),self)
            clf = XGB_Classifier(x_test,trims.loc[subset_indices_test],False,encoder)
            clf.prediction(x_test)
            exp_preds = pr.calc_test_prices(x_test,clf,self,exp_prices)
            preds = best_xgb_reg.predict(pd.concat([x_test,dummy_trims],axis=1).values)
            r2 = r2_score(y_test, preds)
            r2_exp = r2_score(y_test,exp_preds)
            print("R2 Score for straight model predict:", r2)
            evs = explained_variance_score(y_test, exp_preds)
            rmse = np.sqrt(mean_squared_error(y_test, exp_preds))
            print("R2 Score for Exp price calc:", r2_exp)
            print( "EXPLAINED VARIANCE ", evs)
            print("rmse: ",rmse)
        return best_xgb_reg
    
    def prediction(self,feats_data):
        self.preds = self.model.predict(feats_data.values)
        self.preds = pd.Series( self.preds,
                                index=feats_data.index,
                                name=self.target.name    )
        return self.preds
    
    
    