import numpy as np
import pandas as pd

# Suppress convergence warning
import warnings
warnings.simplefilter("ignore")

# Machine Learning
import sklearn
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.svm
import sklearn.preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Plot
import matplotlib.pyplot as plt
from pylab import rcParams
%matplotlib inline
%config InlineBackend.figure_format='retina'
rcParams['figure.figsize'] = 8, 5.5

# Plot heat map of a 2D grid search
def plotGridResults2D(x, y, x_label, y_label, grid_scores):
    
    scores = [s[1] for s in grid_scores]
    scores = np.array(scores).reshape(len(x), len(y))

    plt.figure()
    plt.grid('off')
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.RdYlGn)
    plt.xlabel(y_label)
    plt.ylabel(x_label)
    plt.colorbar()
    plt.xticks(np.arange(len(y)), y, rotation=45)
    plt.yticks(np.arange(len(x)), x)
    plt.title('Validation accuracy')


def plotRoC(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")


# Generate features for a categorical field
def categoricalFeature(series):
    n = float(series.size) # series length
    counts = series.value_counts() # counts of unique values
    
    nbUnique = series.nunique() / 1.0
    hiFreq = counts.max() / n
    loFreq = counts.min() / n
    mdFreq = counts.median() / n
    stdFreq = (counts / n).std()
    argmax = counts.index[0]
    
    return (nbUnique, loFreq, hiFreq, mdFreq, stdFreq, argmax)

         
# Generate features of a numerical field
def numericalFeature(series, withInterval=True):
    series = series.copy()
    series.sort_values(inplace=True)
    
    size = series.size
    min = series.min()
    max = series.max()
    mean = series.mean()
    median = series.median()
    std = series.std()
    range = max - min
    
    if withInterval:
        ints = pd.Series(series[1:].as_matrix() - series[:-1].as_matrix()) if series.size >= 2 else pd.Series([0])
        iMin = ints.min()
        iMax = ints.max()
        iMean = ints.mean()
        iMedian = ints.median()
        iStd = ints.std()
        
        return (size, min, max, mean, median, std, range, iMin, iMax, iMean, iMedian, iStd)
    else:
        (size, min, max, mean, median, std, range)
        

# Generate features for groupby bidder_id and auction.
def numericalFeatureWithGroupBy(data, attr='time', by='auction'):
    # get series and groups
    series = data[attr]
    
    # time stat per bidder
    (size, min, max, _, _, _, range, iMin, iMax, iMean, iMedian, iStd) = numericalFeature(series)
    # total num of bid
    # time of first bid
    # time of last bid
    # range of time
    # min interval between bids
    # max interval between bids
    # avg time between bids
    # median time interval between bids
    # std of time interval between bids
    
    # time stat per auction per bidder
    X = []
    for auction, group in data.groupby(by):
        (gsize, _, _, _, _, _, grange, giMin, giMax, giMean, giMedian, giStd) = numericalFeature(group[attr])
        # num of bids per auction
        # range of time between first and last bids per auction
        # min interval between bids per auction
        # max interval between bids per auction
        # avg time between bids per auction
        # median time interval between bids per auction
        # std of time interval between bids per auction
        
        X.append([gsize, grange, giMin, giMax, giMean, giMedian, giStd])

    X = pd.DataFrame(X, columns=['gsize', 'grange', 'giMin', 'giMax', 'giMean', 'giMedian', 'giStd'])
    
    avgNumBidsPerAuction = X.gsize.mean()
    medNumBidsPerAuction = X.gsize.median()
    stdNumBidsPerAuction = X.gsize.std()
    
    avgRangePerAuction = X.grange.mean()
    medRangePerAuction = X.grange.median()
    stdRangePerAuction = X.grange.std()
    
    avgIntMinPerAuction = X.giMin.mean()
    medIntMinPerAuction = X.giMin.median()
    stdIntMinPerAuction = X.giMin.std()
    avgIntMaxPerAuction = X.giMax.mean()
    medIntMaxPerAuction = X.giMax.median()
    stdIntMaxPerAuction = X.giMax.std()
    
    avgIntMeaPerAuction = X.giMean.mean()
    medIntMeaPerAuction = X.giMean.median()
    stdIntMeaPerAuction = X.giMean.std()
    
    avgIntMedPerAuction = X.giMedian.mean()
    medIntMedPerAuction = X.giMedian.median()
    stdIntMedPerAuction = X.giMedian.std()
    
    avgIntStdPerAuction = X.giStd.mean()
    medIntStdPerAuction = X.giStd.median()
    stdIntStdPerAuction = X.giStd.std()
        
    return (size, min, max, range, iMin, iMax, iMean, iMedian, iStd, # global
            avgNumBidsPerAuction, medNumBidsPerAuction, stdNumBidsPerAuction,
            avgRangePerAuction, medRangePerAuction, stdRangePerAuction,
            avgIntMinPerAuction, medIntMinPerAuction, stdIntMinPerAuction,
            avgIntMaxPerAuction, medIntMaxPerAuction, stdIntMaxPerAuction,
            avgIntMeaPerAuction, medIntMeaPerAuction, stdIntMeaPerAuction,
            avgIntMedPerAuction, medIntMedPerAuction, stdIntMedPerAuction,
            avgIntStdPerAuction, medIntStdPerAuction, stdIntStdPerAuction)
            

def initialize(bids):           
    Xids = []
    X = []
    
    for bidder, group in bids.groupby('bidder_id'):
        # Features for each bidder
        (numUniqIP, loFreqIP, hiFreqIP, mdFreqIP, stdFreqIP, _) = categoricalFeature(group.ip)
        (numUniqDev, loFreqDev, hiFreqDev, mdFreqDev, stdFreqDev, _) = categoricalFeature(group.device)
        (_, _, _, _, _, mer) = categoricalFeature(group.merchandise)
        (numUniqCty, loFreqCty, hiFreqCty, mdFreqCty, stdFreqCty, cty) = categoricalFeature(group.country)
        (numUniqUrl, loFreqUrl, hiFreqUrl, mdFreqUrl, stdFreqUrl, _) = categoricalFeature(group.url)
        (numUniqAuct, loFreqAuct, hiFreqAuct, mdFreqAuct, stdFreqAuct, _) = categoricalFeature(group.auction)
    
        # Features for each bidder, globally and by auction
        (size, min, max, range, iMin, iMax, iMean, iMedian, iStd, # global
        avgNumBidsPerAuction, medNumBidsPerAuction, stdNumBidsPerAuction,
        avgRangePerAuction, medRangePerAuction, stdRangePerAuction,
        avgIntMinPerAuction, medIntMinPerAuction, stdIntMinPerAuction,
        avgIntMaxPerAuction, medIntMaxPerAuction, stdIntMaxPerAuction,
        avgIntMeaPerAuction, medIntMeaPerAuction, stdIntMeaPerAuction,
        avgIntMedPerAuction, medIntMedPerAuction, stdIntMedPerAuction,
        avgIntStdPerAuction, medIntStdPerAuction, stdIntStdPerAuction) = numericalFeatureWithGroupBy(group)
            
        x = [numUniqIP, loFreqIP, hiFreqIP, mdFreqIP, stdFreqIP,
            numUniqDev, loFreqDev, hiFreqDev, mdFreqDev, stdFreqDev,
            mer, # merchandise is largely useless
            numUniqCty, loFreqCty, hiFreqCty, mdFreqCty, stdFreqCty, cty,
            numUniqUrl, loFreqUrl, hiFreqUrl, mdFreqUrl, stdFreqUrl,
            numUniqAuct, loFreqAuct, hiFreqAuct, mdFreqAuct, stdFreqAuct,
            size, min, max, range, iMin, iMax, iMean, iMedian, iStd,
            avgNumBidsPerAuction, medNumBidsPerAuction, stdNumBidsPerAuction,
            avgRangePerAuction, medRangePerAuction, stdRangePerAuction,
            avgIntMinPerAuction, medIntMinPerAuction, stdIntMinPerAuction,
            avgIntMaxPerAuction, medIntMaxPerAuction, stdIntMaxPerAuction,
            avgIntMeaPerAuction, medIntMeaPerAuction, stdIntMeaPerAuction,
            avgIntMedPerAuction, medIntMedPerAuction, stdIntMedPerAuction,
            avgIntStdPerAuction, medIntStdPerAuction, stdIntStdPerAuction]
    
        # Appending result
        Xids.append(bidder)
        X.append(x)
    
    # Features labels
    colNames = ['numUniqIP', 'loFreqIP', 'hiFreqIP', 'mdFreqIP', 'stdFreqIP',
                'numUniqDev', 'loFreqDev', 'hiFreqDev', 'mdFreqDev', 'stdFreqDev',
                'mostCommonMerch',
                'numUniqCty', 'loFreqCty', 'hiFreqCty', 'mdFreqCty', 'stdFreqCty', 'mostCommonCountry',
                'numUniqUrl', 'loFreqUrl', 'hiFreqUrl', 'mdFreqUrl', 'stdFreqUrl',
                'numUniqAuct', 'loFreqAuct', 'hiFreqAuct', 'mdFreqAuct', 'stdFreqAuct',
                'numOfBids', 'firstBidTime', 'lastBidTime', 'bidTimeRange', 'smallestIntBwBids',
                'longestIntBwBids', 'avgIntBwBids', 'medianIntBwBids', 'stdIntBwBids', # global
                'avgNumBidsPerAuction', 'medNumBidsPerAuction', 'stdNumBidsPerAuction',
                'avgRangePerAuction', 'medRangePerAuction', 'stdRangePerAuction',
                'avgIntMinPerAuction', 'medIntMinPerAuction', 'stdIntMinPerAuction',
                'avgIntMaxPerAuction', 'medIntMaxPerAuction', 'stdIntMaxPerAuction',
                'avgIntMeaPerAuction', 'medIntMeaPerAuction', 'stdIntMeaPerAuction',
                'avgIntMedPerAuction', 'medIntMedPerAuction', 'stdIntMedPerAuction',
                'avgIntStdPerAuction', 'medIntStdPerAuction', 'stdIntStdPerAuction']
                
    # Convert to DataFrame
    dataset = pd.DataFrame(X, index=Xids, columns=colNames)
    
    return dataset


def normalizeV2(dataset):
    numeric = dataset.loc[:, dataset.dtypes != 'object']
    mostCommonMerch = dataset.loc[:, 'mostCommonMerch']
    mostCommonCountry = dataset.loc[:, 'mostCommonCountry']
    
    numeric = numeric.apply(lambda x: (x - np.mean(x)) / np.std(x))
    mostCommonMerch = pd.get_dummies(mostCommonMerch)
    mostCommonCountry = pd.get_dummies(mostCommonCountry)
    
    dataset = pd.concat([numeric, mostCommonMerch, mostCommonCountry], axis=1)
    dataset.fillna(0.0, inplace=True)
    
    return dataset
            

# Read the bids (7.6m)
bids = pd.read_csv('bids.csv', header=0)
bids.fillna('-', inplace=True)

# Load train and test bidders lists
train = pd.read_csv('train.csv', header=0, index_col=0)
test = pd.read_csv('test.csv', header=0, index_col=0)

# Join those 2 datasets together (-1 outcome meaning unknown i.e. test)
test['outcome'] = -1.0
bidders = pd.concat((train, test))

dataset = initialize(bids)
#dataset.to_csv('rawFeaturesV2.csv')
datasetN = normalizeV2(dataset)
Labels = (datasetN.join(bidders[['outcome']]))[['outcome']]

# Split learn and final with outcome indices
X_learn = datasetN.loc[(Labels.outcome > -1), :]
y_learn = Labels[(Labels.outcome > -1)]
X_final = datasetN.loc[(Labels == -1).outcome, :]

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X_learn, y_learn, test_size=.25)


############## NB ###############
nbClf = sklearn.naive_bayes.GaussianNB()
nbClf.fit(X_train, y_train.outcome)
y_pred = nbClf.predict(X_test)
print sklearn.metrics.classification_report(y_test.outcome, y_pred)
nfpr, ntpr, _ = roc_curve(y_test, y_pred)
plotRoC(nfpr, ntpr)
#################################

############# SVM ###############

# Coarse grid
C_range = np.r_[np.logspace(-2, 9, 15)]
gamma_range = np.r_[np.logspace(-9, 2, 15)]
gridCoarse = GridSearchCV(sklearn.svm.SVC(C=1.0, kernel='rbf', class_weight='balanced', verbose=False, max_iter=25000),
                    {'C' : C_range, 'gamma': gamma_range},
                   scoring='roc_auc', cv=10, n_jobs=-1)
gridCoarse.fit(X_learn, y_learn.outcome)

C_best = np.round(np.log10(gridCoarse.best_params_['C']))
gamma_best = np.round(np.log10(gridCoarse.best_params_['gamma']))

# Fine grid
Cfine_range = np.r_[np.logspace(C_best - 1, C_best + 1, 15)]
gammafine_range = np.r_[np.logspace(gamma_best - 2, gamma_best + 2, 15)]

gridFine = GridSearchCV(sklearn.svm.SVC(C=1.0, kernel='rbf', class_weight='balanced', verbose=False, max_iter=25000),
                    {'C' : Cfine_range, 'gamma': gammafine_range},
                   scoring='roc_auc', cv=10, n_jobs=-1)
gridFine.fit(X_learn, y_learn.outcome)

svmbestClf = gridFine.best_estimator_
svmbestClf.probability = True

plotGridResults2D(C_range, gamma_range, 'C', 'gamma', gridCoarse.grid_scores_)
plotGridResults2D(Cfine_range, gammafine_range, 'C', 'gamma', gridFine.grid_scores_)

# Fit it
svmbestClf.fit(X_train, y_train.outcome)
y_pred = svmbestClf.predict(X_test)

print sklearn.metrics.classification_report(y_test, y_pred)

# Predict scores
y_score = svmbestClf.predict_proba(X_test)[:,1]

# Plot ROC
sfpr, stpr, _ = roc_curve(y_test, y_score)

plotRoC(sfpr, stpr)

# Fit on learn set and predict final set
svmbestClf.fit(X_learn, y_learn.outcome)
y_final = svmbestClf.predict_proba(X_final)[:,1]
bidders_y_final = pd.DataFrame(np.c_[X_ids[i_Final], y_final], columns=['bidder_id', 'prediction'])
bidders_list = pd.read_csv('test.csv', header=0)
bidders_final = pd.merge(bidders_list[['bidder_id']], bidders_y_final, how='left').fillna(0.0)
bidders_final.to_csv('RBF_SVM_Predictions.csv', index=False)

#################################



######## Random Forest ##########
depth_range = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30, 50, 100])
ntree_range = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000])
grid = GridSearchCV(sklearn.ensemble.RandomForestClassifier(n_estimators=300, max_depth=None,
                                                                   max_features='auto', class_weight='balanced'),
                    {'max_depth' : depth_range,
                    'n_estimators' : ntree_range},
                   cv=sklearn.cross_validation.StratifiedKFold(y_learn.outcome, 10), n_jobs=-1, scoring='roc_auc')
grid.fit(X_learn, y_learn.outcome)

plotGridResults2D(depth_range, ntree_range, 'max depth', 'n estimators', grid.grid_scores_)

rbestClf = grid.best_estimator_
print rbestClf

# Learn on train for test
rbestClf.fit(X_train, y_train.outcome)
y_pred = rbestClf.predict(X_test)

# Classification report
print sklearn.metrics.classification_report(y_test, y_pred)

# Predict scores
y_score = rbestClf.predict_proba(X_test)[:,1]

# ROC
rfpr, rtpr, _ = roc_curve(y_test, y_score)
plotRoC(rfpr, rtpr)


# Feature importance

indices = np.argsort(rbestClf.feature_importances_)[::-1]

# list features and scores
#for f in range(X_train.shape[1]):
#    print("%2d) %-*s %f" % (f+1, 30, X_train.columns[f], bestClf.feature_importances_[indices[f]]))

# plot bar chart
plt.title('Feature Imprtances')
plt.bar(range(10),
       rbestClf.feature_importances_[indices[:10]],
       color='lightblue',
       align='center')
plt.xticks(range(10),
          X_train.columns, rotation=45)
plt.xlim([-1, 10])
plt.tight_layout()
plt.show()


rbestClf.fit(X_learn, y_learn.outcome)
y_final = rbestClf.predict_proba(X_final)[:,1]
bidders_y_final = pd.DataFrame(np.c_[X_ids[i_Final], y_final], columns=['bidder_id', 'prediction'])
bidders_list = pd.read_csv('test.csv', header=0)
bidders_final = pd.merge(bidders_list[['bidder_id']], bidders_y_final, how='left').fillna(0.0)
bidders_final.to_csv('RF_Predictions__.csv', index=False)
#################################

########### ROC Curves ##########
plt.figure()
plt.plot(nfpr, ntpr, label='GNB (area = %0.2f)' % auc(nfpr, ntpr))
plt.plot(sfpr, stpr, label='SVM (area = %0.2f)' % auc(sfpr, stpr))
plt.plot(rfpr, rtpr, label='RF (area = %0.2f)' % auc(rfpr, rtpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.005])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#################################