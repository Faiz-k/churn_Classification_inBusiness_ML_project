from django.shortcuts import render
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



def home(request):
    if request.method == "POST":
        try:
            credit_score = int(request.POST.get('CreditScore'))
            geography = int(request.POST.get('Geography'))
            gender = int(request.POST.get('Gender'))
            age = int(request.POST.get('Age'))
            tenure = int(request.POST.get('Tenure'))
            balance = int(request.POST.get('Balance'))
            num_of_products = int(request.POST.get('NumOfProducts'))
            has_cr_card = int(request.POST.get('HasCrCard'))
            is_active_member = int(request.POST.get('IsActiveMember'))
            estimated_salary = float(request.POST.get('EstimatedSalary'))
        except ValueError:
            return render(request, 'home.html', context={'error': 'Invalid input. Please enter numeric values.'})

        path="C:\\Users\\mf879\\OneDrive\\Desktop\\44_customerchurnClassification\\Churn_Modelling.csv"
        data=pd.read_csv(path)
        categorical_columns = [ 'Geography', 'Gender']
        label_encoders = {}

        for col in categorical_columns:
            le = LabelEncoder()
            data[col + '_new'] = le.fit_transform(data[col])
            label_encoders[col] = le
        inputs=data.drop(['RowNumber','Surname', 'Geography', 'Gender','Exited','CustomerId','Surname'],'columns')
        outputs=data.drop(['RowNumber','CustomerId','Surname','CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Geography_new','Gender_new'],'columns')        
        x_train,x_test,y_train,y_test=train_test_split(inputs,outputs,test_size=0.2)
        model=KNeighborsClassifier(n_neighbors=13)
        model.fit(x_train,y_train)
        y_pred=model.predict([[credit_score,geography,gender,age,tenure,balance ,num_of_products,has_cr_card,is_active_member,estimated_salary]])

        return render(request, "home.html", context={'y_pred':y_pred })

    return render(request, 'home.html')


