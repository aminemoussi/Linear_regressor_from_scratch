import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import linear_regression as lr

    #DATA pre-processing
data = pd.read_csv("salary_data.csv")

#print(data.head())
#print(data.shape)
#print(data.isnull().sum())      #nothing missing

   #deviding the data
x = data.iloc[:, :-1].values          #extracts all rows from all columns except the last one
y = data.iloc[:, 1].values            # .values to get it in the form of a numpyarray wich can facilitate mathmatecal operations
#print(x)
#print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=2)

    #training the model
salary_model = lr.linear_regression(learning_rate=.01, iterations=10000)
loss_history = []
salary_model.fit(x_train, y_train)

        #model's parameters
#print("for optimal results: ")
#print("Weight= ", salary_model.weights)
#print("Bias= ", salary_model.bias)

y_predicted = salary_model.predict(x_test)

#print(y_predicted)

plt.scatter(x_test, y_test, color = "red")
plt.plot(x_test, y_predicted, color = "blue")
plt.show()

score = metrics.r2_score(y_test, y_predicted)

print("the accuracy =", score*100)

