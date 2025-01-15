from sklearn.metrics import (confusion_matrix)

from sklearn.metrics import recall_score, accuracy_score, precision_score,f1_score

def calculate_metrics (y_true, y_pred):
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    specificity = recall_score(y_true=y_true, y_pred=y_pred,pos_label=0)
    return f1,accuracy,precision,recall,specificity



ytrue = [True, False, True, True, False, False, False, False, True, True]
ypred = [False, False, True, True, False, False, True, False, True, False]

f1,accuracy,precision,recall,specificity =calculate_metrics(y_true=ytrue, y_pred=ypred)

print('f1 =  ', f1)
print('accuracy =  ', accuracy)
print('precision =  ', precision)
print('recall =  ', recall)
print('specificity = ', specificity)

''''
specificity = recall_score(ytrue, ypred, pos_label=0)
recall = recall_score(ytrue, ypred )
 
print("Specificity or True Negative Rate: ", specificity)
print("RECALL      or True Negative Rate: ", recall)

tn, fp, fn, tp = confusion_matrix(ytrue, ypred).ravel()
specificity = tn / (tn+fp)
print("Specificity or True Negative Rate: ", specificity)
'''


