import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBClassifier

print "Importing dataset"
train = pd.read_csv('new_x.csv')
test = pd.read_csv('new_x_test.csv')
train_stem = pd.read_csv('stemmed_doc.csv')
test_stem = pd.read_csv('stemmed_doc_test.csv')
train_stem = pd.DataFrame(train_stem.stem_desc_x + " " + train_stem.stem_desc_y + " " + train_stem.stem_title_x + " " + train_stem.stem_title_y, columns = ['combined'])
test_stem = pd.DataFrame(test_stem.stem_desc_x + " " + test_stem.stem_desc_y + " " + test_stem.stem_title_x + " " + test_stem.stem_title_y, columns = ['combined'])
train_stem.combined = train_stem.combined.map(lambda x : '' if pd.isnull(x) else x)
test_stem.combined = test_stem.combined.map(lambda x : '' if pd.isnull(x) else x)
train = pd.concat([train, train_stem.combined], axis = 1)
test = pd.concat([test, test_stem.combined], axis = 1)

class parCatIDTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.parentCategoryID_x)

    def fit(self, X, y=None, **fit_params):
        return self
    
class priceDiffTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.price_diff, columns = ['price_diff'])

    def fit(self, X, y=None, **fit_params):
        return self

class priceMeanTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.price_mean, columns = ['price_mean'])

    def fit(self, X, y=None, **fit_params):
        return self

class imagesDiffTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.images_num_diff, columns = ['images_num_diff'])

    def fit(self, X, y=None, **fit_params):
        return self

class sameMetroTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.same_metro, columns = ['same_metro'])

    def fit(self, X, y=None, **fit_params):
        return self

class titleXTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.title_x_len, columns = ['title_x_len'])

    def fit(self, X, y=None, **fit_params):
        return self

class titleYTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.title_y_len, columns = ['title_y_len'])

    def fit(self, X, y=None, **fit_params):
        return self

class descXTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.desc_x_len, columns = ['desc_x_len'])

    def fit(self, X, y=None, **fit_params):
        return self

class descYTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.desc_y_len, columns = ['desc_y_len'])

    def fit(self, X, y=None, **fit_params):
        return self

class titleDiffTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.title_len_diff, columns = ['title_len_diff'])

    def fit(self, X, y=None, **fit_params):
        return self

class descDiffTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.desc_len_diff, columns = ['desc_len_diff'])

    def fit(self, X, y=None, **fit_params):
        return self

class titleXUTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.title_x_unique_len, columns = ['title_x_unique_len'])

    def fit(self, X, y=None, **fit_params):
        return self

class titleYUTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.title_y_unique_len, columns = ['title_y_unique_len'])

    def fit(self, X, y=None, **fit_params):
        return self

class descXUTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.desc_x_unique_len, columns = ['desc_x_unique_len'])

    def fit(self, X, y=None, **fit_params):
        return self

class descYUTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.desc_y_unique_len, columns = ['desc_y_unique_len'])

    def fit(self, X, y=None, **fit_params):
        return self


class ContentTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return X.combined

    def fit(self, X, y=None, **fit_params):
        return self

count_vect = CountVectorizer()
log_reg = LogisticRegression()

pipe_log = Pipeline([
    ('features', FeatureUnion([
        ('pipe_cont', Pipeline([
            ('content', ContentTransformer()),
            ('count_vect', CountVectorizer(ngram_range = (1, 2)))
        ])),
        ('price_diff', priceDiffTransformer()), 
        ('parentCategoryID_x', parCatIDTransformer()),
        ('price_mean', priceMeanTransformer()),
        ('images_diff', imagesDiffTransformer()),
        ('same_metro', sameMetroTransformer()),
        ('title_x_len', titleXTransformer()),
        ('title_y_len', titleYTransformer()),
        ('desc_x_len', descXTransformer()),
        ('desc_y_len', descYTransformer()),
        ('title_diff', titleDiffTransformer()),
        ('desc_diff', descDiffTransformer()),
        ('title_x_unique', titleXUTransformer()),
        ('title_y_unique', titleYUTransformer()),
        ('desc_x_unique', descXUTransformer()),
        ('desc_y_unique', descYUTransformer())
    ])),
    ('log_reg', log_reg)])

print "Training"
pipe_log.fit(train.iloc[:, train.columns != 'isDuplicate'], train.isDuplicate)
print "Predicting"
pred_train = pipe_log.predict(train.iloc[:, train.columns != 'isDuplicate'])
print "Train Accuracy:", accuracy_score(train.isDuplicate, pred_train)
pred_test_prob = pipe_log.predict_proba(test)
pd.DataFrame(pred_test_prob[:,1], columns = ['probability']).to_csv('simple.csv')