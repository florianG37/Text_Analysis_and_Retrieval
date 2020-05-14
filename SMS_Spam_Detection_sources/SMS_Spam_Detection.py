import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


## Loading Data

dataSet = pandas.read_csv("SMSSpamCollection.txt", sep="	", header=None)
dataSet.columns = ["label", "SMS"]


## Spliting Data

tab_SMS = dataSet["SMS"]
tab_label = dataSet["label"]

## Pre_process the Data
p = tab_SMS.str.replace(r'£|\$', 'money-symbol')
p = p.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phone-number')

##Feature Engineering

#trigram

#bigram

ham_bigram=['lt gt','gon na', 'call later', 'let know', 'sorry call', 'r u','u r', 'good morning','take care','u wan', 'wan na','lt decimal','decimal gt','new year', 'u get','pls send','ok lor','gt lt','u still','good night']
for w in ham_bigram:
    p[p.str.contains(w)]= p[p.str.contains(w)] + " Bigram--HAM"

spam_bigram=['please call','po box', 'guaranteed call', 'prize guaranteed', 'call landline', 'selected receive','contact u', 'send stop','every week','await collection', 'call claim','urgent mobile','call land','land line', 'customer service','chance win','free entry','claim call','private account','account, statement']
for w in spam_bigram:
    p[p.str.contains(w)]= p[p.str.contains(w)] + " spamBigram"

#Train Data
tab_SMS_train = p[0:4800]
tab_label_train = tab_label[0:4800]

#Test Data
tab_SMS_test = p[4800:]
tab_label_test = tab_label[4800:]

#Extract Features
cv = CountVectorizer()
features = cv.fit_transform(tab_SMS_train)

#Build the model
model = MultinomialNB();
model.fit(features, tab_label_train)


#Testing performance
test = cv.transform(tab_SMS_test)
print("Accuracy of the model is ", model.score(test, tab_label_test))



