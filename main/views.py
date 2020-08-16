from django.http import HttpResponse
from django.shortcuts import render


from django.shortcuts import render
from django.http import HttpResponse
import matplotlib.pyplot as plt
import io
import urllib, base64
import requests
import datetime
import numpy as np
# from scripy.interpolate import spline
from datetime import date
import holidays
import requests
import datetime
import numpy as np
# from scripy.interpolate import spline
from datetime import date
import holidays
import requests
import datetime
import matplotlib.pyplot as plt
import numpy as np
# from scripy.interpolate import spline
from datetime import date
from pandas import DataFrame
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
# from django.shortcuts import render_to_response
import mpld3
from django.shortcuts import render
from django.http import JsonResponse

def index(request):
    # Create your views here.


    return render(request, 'index.html')



def contact(request):
    # Create your views here.


    return render(request, 'contact.html')


def about(request):
    return render(request, 'about.html')


def webpage3(request):

# Select country
    us_holidays = holidays.US()
    holidaylist=[]
    for ptr in holidays.US(years = 2020).items():


        ptr = list(ptr)
        print(ptr)
        print("holidays")

        holidaylist.append(ptr[0].strftime('%Y-%m-%d'))
        print(holidaylist)



    def NormalModel():

        today=datetime.date.today()
        today1=today.strftime('%Y-%m-%d')
        query=request.GET.get('name')
        print(query)
        result = str(query)
        print(result)

        # url=('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NYSE:GOOGL&apikey=XCIMZL4S81DCVOQO')
        url1='https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='
        url2=result
        url3='&apikey=XCIMZL4S81DCVOQO'
        url=url1+url2+url3

        # url=('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=GOOGL&apikey=XCIMZL4S81DCVOQO')
        response=requests.get(url)
        data=response.json()
        xlist=[]
        ylist=[]
        counter=32
        while (counter>0):
            yesterday1=today-datetime.timedelta(days=counter)

            if(yesterday1.weekday()==5 or yesterday1.weekday()==6):
                xlist.append(yesterday1)
                xlist.remove(yesterday1)

            else:
                yesterday1=yesterday1.strftime('%Y-%m-%d')
                xlist.append(yesterday1)
                holidaycount=0
                while (holidaycount<len(holidaylist)):

                    if (yesterday1==holidaylist[holidaycount]):
                        xlist.remove(yesterday1)
                    holidaycount=holidaycount+1

            counter=counter-1

        index=0
        while(index< len(xlist)):
            print(data['Time Series (Daily)'][xlist[index]]['4. close'])

            ylist.append(data['Time Series (Daily)'][xlist[index]]['4. close'])
            index=index+1


        new_list = []
        for item in ylist:
            new_list.append(float(item))

        print("newlist")

        print(new_list)
        df=DataFrame(new_list, columns=['stock_prices'])

        values=df.values
        print(values)

        values.size

        train=values[0:]
        import pmdarima
        from pmdarima import auto_arima
        auto_model = auto_arima(train, start_p=1, start_q=1,
                         test='adf',       # use adftest to find optimal 'd'
                         max_p=3, max_q=3, # maximum p and q
                         m=1,              # frequency of series
                         d=None,           # let model determine 'd'
                         seasonal=False,   # No Seasonality
                          start_P=0,
                         D=0,
                          trace=True,
                          error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)

        auto_model.summary()





        from statsmodels.tsa.stattools import adfuller
        def adf_test(timeseries):
        #Perform Dickey-Fuller test:
            print ('Results of Dickey-Fuller Test:')
            dftest = adfuller(timeseries, autolag='AIC')
            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
            for key,value in dftest[4].items():
               dfoutput['Critical Value (%s)'%key] = value
            print (dfoutput)



        adf_test(df['stock_prices'])
        print("hi")

        from statsmodels.tsa.stattools import adfuller
        df2=adfuller(df['stock_prices'],autolag='AIC')
        df_out=pd.Series(df2[0:4], index=['test statistic', 'pvalue', 'number of lags used', 'number of obs'])

        print(df_out)



        from statsmodels.tsa.stattools import kpss
    #define KPSS
        # def kpss_test(timeseries):
        #     print ('Results of KPSS Test:')
        #     kpsstest = kpss(timeseries, regression='c')
        #     kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
        #     for key,value in kpsstest[3].items():
        #         kpss_output['Critical Value (%s)'%key] = value
        #     print (kpss_output)
        #
        #
        # kpss_test(df['stock_prices'])


        counter=0
        xlist2=[]
        while(counter<21):
            dates2=datetime.date.today()+datetime.timedelta(days=counter)

            if(dates2.weekday()==5 or dates2.weekday()==6):
                xlist2.append(dates2.strftime('%Y-%m-%d'))
                xlist2.remove(dates2.strftime('%Y-%m-%d'))


            else:
                dates2=dates2.strftime('%Y-%m-%d')
                xlist2.append(dates2)
                holidaycount=0
                while (holidaycount<len(holidaylist)):

                    if (dates2==holidaylist[holidaycount]):
                        xlist2.remove(dates2)
                        # print("removed" +dates2)
                    holidaycount=holidaycount+1

            counter=counter+1



        # plt.show()
        print("predictions")

        print (ylist)
        print (len(ylist))
        print (len(xlist))
        print(xlist)
        print("xlist")

        print(xlist)

        test=values[12:]
        print(test)
        print("other")
        plt.plot(train)
        plt.plot(test, color='orange')
        # plt.show()
        predictions=[]


        test2=values[10:]
        print(test2)
        xlist.extend(xlist2)
        print(xlist)
        model_ar=AR(test2)
        model_ar_fit=model_ar.fit()
        print("p")
        predictions=model_ar_fit.predict(start=len(test2), end=len(test2)+10)
        new_list.extend(predictions)
        print(new_list)
        print(xlist[13:])


        print(predictions)
        print("predictions!")

        # percent error
        percentlist=[]
        exact=values[10:21]
        value=0
        while (value<len(predictions)):
            subtract=predictions[value] - exact[value]
            # print(subtract)
            error=(subtract)/(exact[value])
            # print(error)
            percent = abs(error) * 100
            print(percent)
            value=value+1
            percentlist.append(percent)
        print(percentlist)
        print("percent")
        i=0
        sum=0
        while (i<len(percentlist)):
            sum=sum+percentlist[i]
            # print(sum)
            i=i+1
        average=(sum)/(len(percentlist))
        print(average)
        print("average")


        print(xlist[12:25])
        data_tuples = list(zip(xlist[9:36],new_list[9:]))
        df1=pd.DataFrame(data_tuples, columns=['Date','Value'])
        print(df1)
        print("data")
        #define cutoff date
        fig, ax = plt.subplots(figsize=(12, 4))
        dateforgraph=datetime.date.today()

        cutoff =dateforgraph.strftime('%Y-%m-%d')
        #sort dataframe because unsorted dates will not plot properly
        df1 = df1.sort_values(["Date"])
        #plot the whole dataframe in yellow
        plt.plot(df1.Date, df1.Value, c = "b", label = "before {}".format(cutoff))
        #plot the conditional data on top in red


        plt.plot(df1[df1.Date >= cutoff].Date, df1[df1.Date >= cutoff].Value, c = "y", label = "after {}".format(cutoff))
        plt.gcf().autofmt_xdate()
        plt.xticks(rotation = 45)


        plt.legend()
        # plt.show()

        return predictions

    def LogModel():

        today=datetime.date.today()
        today1=today.strftime('%Y-%m-%d')
        # url=('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NYSE:GOOGL&apikey=XCIMZL4S81DCVOQO')
        query=request.GET.get('name')
        print(query)
        result = str(query)
        print(result)

        # url=('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NYSE:GOOGL&apikey=XCIMZL4S81DCVOQO')
        url1='https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='
        url2=result
        url3='&apikey=XCIMZL4S81DCVOQO'
        url=url1+url2+url3

        response=requests.get(url)
        data=response.json()
        print(data)
        xlist=[]
        ylist=[]
        counter=32
        while (counter>0):
            yesterday1=today-datetime.timedelta(days=counter)
            #

            if(yesterday1.weekday()==5 or yesterday1.weekday()==6):
                xlist.append(yesterday1)
                xlist.remove(yesterday1)
                print(yesterday1)
                print("removed")

            else:
                yesterday1=yesterday1.strftime('%Y-%m-%d')
                xlist.append(yesterday1)
                holidaycount=0
                while (holidaycount<len(holidaylist)):

                    if (yesterday1==holidaylist[holidaycount]):
                        xlist.remove(yesterday1)
                        print("removed" +yesterday1)
                    holidaycount=holidaycount+1

            counter=counter-1
            print(counter)


        # print(date)

        index=0
        while(index< len(xlist)):
            # print(data['Time Series (Daily)'][xlist[index]]['4. close'])
            #print(data.head())
            ylist.append(data['Time Series (Daily)'][xlist[index]]['4. close'])
            index=index+1
            print(index)




            # list1 = sorted(ylist, key=float)

            # counter=0
            # while (counter<len(ylist)):
            #
            #     ylist[counter]=float(ylist[counter])
            #
            #
            # counter=counter+1

        new_list = []
        for item in ylist:
            new_list.append(float(item))

        df=DataFrame(new_list, columns=['stock_prices'])
        # print(df)
        print(df[0:10])
        print("df")
        values=df.values
        plt.plot(df)
        print(values)
        print("values")
        values.size
        print(values.size)
        print("size")
        train=values[0:]



    #Auto_arima
        import pmdarima
        from pmdarima import auto_arima
        auto_model = auto_arima(train, start_p=1, start_q=1,
                         test='adf',       # use adftest to find optimal 'd'
                         max_p=3, max_q=3, # maximum p and q
                         m=1,              # frequency of series
                         d=None,           # let model determine 'd'
                         seasonal=False,   # No Seasonality
                          start_P=0,
                         D=0,
                          trace=True,
                          error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)

        auto_model.summary()




    #ADF Test
        from statsmodels.tsa.stattools import adfuller
        def adf_test(timeseries):
        #Perform Dickey-Fuller test:
            print ('Results of Dickey-Fuller Test:')
            dftest = adfuller(timeseries, autolag='AIC')
            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
            for key,value in dftest[4].items():
               dfoutput['Critical Value (%s)'%key] = value
            print (dfoutput)
        # df_log=np.log(df)


        adf_test(df['stock_prices'])
        # plt.plot(df_log)
        # df_log_dif=df_log-df_log.shift()
        # df_log_dif.dropna(inplace=True)
        from statsmodels.tsa.stattools import adfuller
        df2=adfuller(df['stock_prices'],autolag='AIC')
        df_out=pd.Series(df2[0:4], index=['test statistic', 'pvalue', 'number of lags used', 'number of obs'])

        print(df_out)



    #Logarithmic Differencing




        import math
        index=0
        numberlist=[]
        datavalues=df.values[10:]

        while (index<len(datavalues)):
            number=math.log(datavalues[index],10)
            numberlist.append(number)
            index=index+1
        print(numberlist)
        adf_test(numberlist)
        print("s?")



        from statsmodels.tsa.stattools import adfuller
        def adf_test(timeseries):
        #Perform Dickey-Fuller test:
            print ('Results of Dickey-Fuller Test:')
            dftest = adfuller(timeseries, autolag='AIC')
            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
            for key,value in dftest[4].items():
               dfoutput['Critical Value (%s)'%key] = value
            print (dfoutput)




        import pmdarima
        from pmdarima import auto_arima
        auto_model = auto_arima(numberlist, start_p=1, start_q=1,
                         test='adf',       # use adftest to find optimal 'd'
                         max_p=3, max_q=3, # maximum p and q
                         m=1,              # frequency of series
                         d=None,           # let model determine 'd'
                         seasonal=False,   # No Seasonality
                          start_P=0,
                         D=0,
                          trace=True,
                          error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)


        model_arima = ARIMA(numberlist, order=(0, 1, 0))
        model_arima_fit=model_arima.fit()
        predictions2=model_arima_fit.forecast(steps=10)[0]
        print(predictions2)


        model_ar=AR(numberlist)
        model_ar_fit=model_ar.fit()
        print("p")
        predictions=model_ar_fit.predict(start=len(numberlist), end=len(numberlist)+10)
        index1=0
        list=[]
        while (index1<len(predictions)):
            exponent=pow(10, predictions[index1])
            list.append(exponent)
            index1=index1+1

        print(list)


        plt.plot(list)
        # plt.show()
        print("forecast")




        percentlist=[]
        exact=df.values[10:21]
        value=0
        while (value<len(list)):
            subtract=list[value] - exact[value]
            # print(subtract)
            error=(subtract)/(exact[value])
            # print(error)
            percent = abs(error) * 100
            print(percent)
            value=value+1
            percentlist.append(percent)
        print(percentlist)
        print("percent")
        i=0
        sum=0
        while (i<len(percentlist)):
            sum=sum+percentlist[i]
            # print(sum)
            i=i+1
        average=(sum)/(len(percentlist))
        print(average)
        print("average")


        print(np.expm1(predictions2.cumsum()))
        exp= np.exp(predictions2)
        print(exp)
        print("ARIMAFINAL")



        def inverse_difference(history, yhat, interval=1):
    	       return yhat + history[-interval]






        return list



    def DifferencesModel():

        today=datetime.date.today()
        today1=today.strftime('%Y-%m-%d')
        query=request.GET.get('name')
        print(query)
        result = str(query)
        print(result)

        # url=('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NYSE:GOOGL&apikey=XCIMZL4S81DCVOQO')
        url1='https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='
        url2=result
        url3='&apikey=XCIMZL4S81DCVOQO'
        url=url1+url2+url3
        # url=('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NYSE:GOOGL&apikey=XCIMZL4S81DCVOQO')
        data=requests.get(url).json()
        # data=response.json()
        # print(data)
        xlist=[]
        ylist=[]
        counter=32
        while (counter>0):
            yesterday1=today-datetime.timedelta(days=counter)
            #

            if(yesterday1.weekday()==5 or yesterday1.weekday()==6):
                xlist.append(yesterday1)
                xlist.remove(yesterday1)
                # print(yesterday1)
                # print("removed")

            else:
                yesterday1=yesterday1.strftime('%Y-%m-%d')
                xlist.append(yesterday1)
                holidaycount=0
                while (holidaycount<len(holidaylist)):

                    if (yesterday1==holidaylist[holidaycount]):
                        xlist.remove(yesterday1)
                        print("removed" +yesterday1)
                    holidaycount=holidaycount+1

            counter=counter-1
            # print(counter)


        # print(date)

        index=0
        while(index< len(xlist)):
            # print(data['Time Series (Daily)'][xlist[index]]['4. close'])
            #print(data.head())
            ylist.append(data['Time Series (Daily)'][xlist[index]]['4. close'])
            index=index+1
            # print(index)


        new_list = []
        for item in ylist:
            new_list.append(float(item))

        # print("newlist")

        # print(new_list)
        df=DataFrame(new_list, columns=['stock_prices'])
        # print(df)
        values=df.values
        print(values)
        print("values")
        values.size
        # print(values.size)
        # print("size")
        train=values[0:]
        import pmdarima
        from pmdarima import auto_arima
        auto_model = auto_arima(train, start_p=1, start_q=1,
                         test='adf',       # use adftest to find optimal 'd'
                         max_p=3, max_q=3, # maximum p and q
                         m=1,              # frequency of series
                         d=None,           # let model determine 'd'
                         seasonal=False,   # No Seasonality
                          start_P=0,
                         D=0,
                          trace=True,
                          error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)

        auto_model.summary()





        from statsmodels.tsa.stattools import adfuller
        def adf_test(timeseries):
        #Perform Dickey-Fuller test:
            print ('Results of Dickey-Fuller Test:')
            dftest = adfuller(timeseries, autolag='AIC')
            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
            for key,value in dftest[4].items():
               dfoutput['Critical Value (%s)'%key] = value
            print (dfoutput)
        # df_log=np.log(df)


        adf_test(df['stock_prices'])
        # plt.plot(df_log)
        # df_log_dif=df_log-df_log.shift()
        # df_log_dif.dropna(inplace=True)
        from statsmodels.tsa.stattools import adfuller
        df2=adfuller(df['stock_prices'],autolag='AIC')
        df_out=pd.Series(df2[0:4], index=['test statistic', 'pvalue', 'number of lags used', 'number of obs'])

        print(df_out)












        # difference dataset
        def difference(data, interval):
        	return [data[i] - data[i - interval] for i in range(interval, len(data))]

        # invert difference
        def invert_difference(orig_data, diff_data, interval):
        	return [diff_data[i-interval] + orig_data[i-interval] for i in range(interval, len(orig_data))]

        # define dataset
        # data = [x for x in range(1, 10)]
        # print(data)
        # difference transform
        transformed = difference(df['stock_prices'], 1)
        print(transformed)
        # print("difference")
        print( df['stock_prices'][0:10])




        def inverse_difference(last_ob, value):
    	       return value + last_ob



        # inverted1 = [inverse_difference(df['stock_prices'][i], transformed[i]) for i in range(len(transformed))]
        # print(inverted1)



        from statsmodels.tsa.stattools import adfuller
        def adf_test(timeseries):
        #Perform Dickey-Fuller test:
            print ('Results of Dickey-Fuller Test:')
            dftest = adfuller(timeseries, autolag='AIC')
            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
            for key,value in dftest[4].items():
               dfoutput['Critical Value (%s)'%key] = value
            print (dfoutput)


        from statsmodels.tsa.stattools import adfuller
        df2=adfuller(df['stock_prices'],autolag='AIC')
        df_out=pd.Series(df2[0:4], index=['test statistic', 'pvalue', 'number of lags used', 'number of obs'])

        print(df_out)





        # invert difference


        test3=transformed
        # print(test3)
        # print("test3")
        # xlist.extend(xlist2)
        # print(xlist)
        model_ar=AR(test3)
        model_ar_fit=model_ar.fit()
        # print("p")
        predictions=model_ar_fit.predict(start=len(test3), end=len(test3)+10)
        inverted = [inverse_difference(df['stock_prices'][i], predictions[i]) for i in range(len(predictions))]
        print(inverted)
        print("inverted values2")
        new_list.extend(inverted)

        counter=0
        xlist2=[]
        while(counter<21):
            dates2=datetime.date.today()+datetime.timedelta(days=counter)
            # print(dates2)
            # print("dates2")
            if(dates2.weekday()==5 or dates2.weekday()==6):
                xlist2.append(dates2.strftime('%Y-%m-%d'))
                xlist2.remove(dates2.strftime('%Y-%m-%d'))
                # print(dates2)
                # print("removed")

            else:
                dates2=dates2.strftime('%Y-%m-%d')
                xlist2.append(dates2)
                holidaycount=0
                while (holidaycount<len(holidaylist)):

                    if (dates2==holidaylist[holidaycount]):
                        xlist2.remove(dates2)
                        # print("removed" +dates2)
                    holidaycount=holidaycount+1

            counter=counter+1
            # print(counter)
            # print(xlist2)



        xlist.extend(xlist2)
        # print(xlist)
        # print("list")
        print(inverted)
        data_tuples = list(zip(xlist[9:36],new_list[9:30]))
        df2=pd.DataFrame(data_tuples, columns=['Date','Value'])

        dateforgraph=datetime.date.today()

        cutoff =dateforgraph.strftime('%Y-%m-%d')
        #sort dataframe because unsorted dates will not plot properly
        df2 = df2.sort_values(["Date"])
        print(df2)
        #plot the whole dataframe in yellow
        plt.plot(df2.Date, df2.Value, c = "orange", label = "before {}".format(cutoff))
        #plot the conditional data on top in red
        plt.plot(df2[df2.Date >= cutoff].Date, df2[df2.Date >= cutoff].Value, c = "b", label = "after {}".format(cutoff))
        plt.gcf().autofmt_xdate()
        plt.xticks(rotation = 45)
        plt.legend()
        # plt.show()








        print("inverted")
        # new_list.extend(predictions)
        print(new_list)
        # print(xlist[13:])
        from statsmodels.tsa.stattools import kpss
    #define KPSS
        def kpss_test(timeseries):
            print ('Results of KPSS Test:')
            kpsstest = kpss(timeseries, regression='c')
            kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
            for key,value in kpsstest[3].items():
                kpss_output['Critical Value (%s)'%key] = value
            print (kpss_output)


        kpss_test(df['stock_prices'])


        counter=0
        xlist2=[]
        while(counter<21):
            dates2=datetime.date.today()+datetime.timedelta(days=counter)
            # print(dates2)
            # print("dates2")
            if(dates2.weekday()==5 or dates2.weekday()==6):
                xlist2.append(dates2.strftime('%Y-%m-%d'))
                xlist2.remove(dates2.strftime('%Y-%m-%d'))
                # print(dates2)
                # print("removed")

            else:
                dates2=dates2.strftime('%Y-%m-%d')
                xlist2.append(dates2)
                holidaycount=0
                while (holidaycount<len(holidaylist)):

                    if (dates2==holidaylist[holidaycount]):
                        xlist2.remove(dates2)
                        # print("removed" +dates2)
                    holidaycount=holidaycount+1

            counter=counter+1
            # print(counter)
            # print(xlist2)



        test=values[12:]





        import numpy

        def difference(dataset, interval=1):
        	diff = list()
        	for i in range(interval, len(dataset)):
        		value = dataset[i] - dataset[i - interval]
        		diff.append(value)
        	return numpy.array(diff)

    # invert differenced value
        def inverse_difference(history, yhat, interval=1):
    	       return yhat + history[-interval]


    # seasonal difference
        X = df.values
        print(X[0:10])
        print("10 values")
        differenced = difference(X[10:])
        print(differenced)
        from statsmodels.tsa.stattools import adfuller
        differenced2=differenced[0:5]
        def adf_test(timeseries):
        #Perform Dickey-Fuller test:
            print ('Results of Dickey-Fuller Test of difference data:')
            dftest = adfuller(timeseries, autolag='AIC')
            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
            for key,value in dftest[4].items():
               dfoutput['Critical Value (%s)'%key] = value
            print (dfoutput)
        # df_log=np.log(df)


        adf_test(differenced2)


        model = ARIMA(differenced, order=(1,0,0))
        model_fit = model.fit(disp=0)
        # multi-step out-of-sample forecast
        start_index = len(differenced)
        end_index = start_index + 10
        forecast = model_fit.predict(start=start_index, end=end_index)

        # invert the differenced forecast to something usable
        history = [x for x in X[0:10]]
        day = 1
        invertedlist=[]
        for yhat in forecast:
            invertedlist.append(inverse_difference(history, yhat))


            day += 1

        # print(history)
        print(invertedlist)
        npa = np.asarray(invertedlist, dtype=np.float32)
        print(npa)
        exact=np.asarray(X[10:21], dtype=np.float32)
        print(exact)
        # percent error
        percentlist=[]
        value=0
        while (value<len(npa)):
            subtract=npa[value] - exact[value]
            # print(subtract)
            error=(subtract)/(exact[value])
            # print(error)
            percent = abs(error) * 100
            print(percent)
            value=value+1
            percentlist.append(percent)
        print(percentlist)
        i=0
        sum=0
        while (i<len(percentlist)):
            sum=sum+percentlist[i]
            # print(sum)
            i=i+1
        average=(sum)/(len(percentlist))
        print(average)
        print("average")

        print("history")

        new_list.extend(history)
        # plt.plot(xlist,new_list, color='blue')
        data_tuples = list(zip(xlist,new_list))
        df1=pd.DataFrame(data_tuples, columns=['Date','Value'])
        print(df1)
        print("data")
        #define cutoff date
        dateforgraph=datetime.date.today()

        cutoff =dateforgraph.strftime('%Y-%m-%d')
        #sort dataframe because unsorted dates will not plot properly
        df1 = df1.sort_values(["Date"])
        #plot the whole dataframe in yellow
        plt.plot(df1.Date, df1.Value, c = "b", label = "before {}".format(cutoff))
        #plot the conditional data on top in red
        plt.plot(df1[df1.Date >= cutoff].Date, df1[df1.Date >= cutoff].Value, c = "r", label = "after {}".format(cutoff))
        plt.gcf().autofmt_xdate()
        plt.xticks(rotation = 45)
        plt.legend()
        # plt.show()

        return npa











    result1=NormalModel()
    print(result1)

    ratio=(92.66171857)/(280.000691)

    indexnormal=0
    productlistnormal=[]
    while (indexnormal<len(result1)):
        product=(ratio)*(result1[indexnormal])
        productlistnormal.append(product)
        indexnormal=indexnormal+1

    print(productlistnormal)
    print("productlistnormal")

    print("result1 from first model")
    result2=LogModel()
    print(result2)
    import numpy
    result2=numpy.concatenate([result2], axis=0)

    ratiolog=(92.94373087)/(280.000691)

    indexlog=0
    productlistlog=[]
    while (indexlog<len(result2)):
        productlog=(ratio)*(result2[indexlog])
        productlistlog.append(productlog)
        indexlog=indexlog+1

    print(productlistlog)
    print("productlistlog")

    print("result2 from second model")
    result3=DifferencesModel()
    print(result3)

    ratiodiff=(94.39524154)/(280.000691)
    import numpy
    indexdiff=0
    result4=numpy.concatenate(result3, axis=0)
    productlistdiff=[]
    while (indexdiff<len(result4)):
        productdiff=(ratiodiff)*(result4[indexdiff])
        productlistdiff.append(productdiff)
        indexdiff=indexdiff+1

    print(productlistdiff)
    print("productlistdiff")


    print("result3 from third model")



    #final values list


    allvalueslist=[]

    indexallvalues=0

    while(indexallvalues<len(productlistdiff)):
        sumallvalues=productlistdiff[indexallvalues]+productlistlog[indexallvalues]+productlistnormal[indexallvalues]
        print(sumallvalues)
        allvalueslist.append(sumallvalues)

        indexallvalues=indexallvalues+1

    print(allvalueslist)
    print("allvalueslist")


    today=datetime.date.today()
    today1=today.strftime('%Y-%m-%d')
    query=request.GET.get('name')
    print(query)
    result = str(query)
    print(result)

    # url=('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NYSE:GOOGL&apikey=XCIMZL4S81DCVOQO')
    url1='https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='
    url2=result
    url3='&apikey=XCIMZL4S81DCVOQO'
    url=url1+url2+url3
    # url=('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NYSE:GOOGL&apikey=XCIMZL4S81DCVOQO')
    data=requests.get(url).json()
    # data=response.json()
    # print(data)
    xlist=[]
    ylist=[]
    counter=32
    while (counter>0):
        yesterday1=today-datetime.timedelta(days=counter)
        #

        if(yesterday1.weekday()==5 or yesterday1.weekday()==6):
            xlist.append(yesterday1)
            xlist.remove(yesterday1)
            # print(yesterday1)
            # print("removed")

        else:
            yesterday1=yesterday1.strftime('%Y-%m-%d')
            xlist.append(yesterday1)
            holidaycount=0
            while (holidaycount<len(holidaylist)):

                if (yesterday1==holidaylist[holidaycount]):
                    xlist.remove(yesterday1)
                    print("removed" +yesterday1)
                holidaycount=holidaycount+1

        counter=counter-1
        # print(counter)


    # print(date)

    index=0
    while(index< len(xlist)):
        # print(data['Time Series (Daily)'][xlist[index]]['4. close'])
        #print(data.head())
        ylist.append(data['Time Series (Daily)'][xlist[index]]['4. close'])
        index=index+1
        # print(index)




        # list1 = sorted(ylist, key=float)

        # counter=0
        # while (counter<len(ylist)):
        #
        #     ylist[counter]=float(ylist[counter])
        #
        #
        # counter=counter+1

    new_list = []
    for item in ylist:
        new_list.append(float(item))







    counter=0
    xlist2=[]
    while(counter<21):
        dates2=datetime.date.today()+datetime.timedelta(days=counter)
        # print(dates2)
        # print("dates2")
        if(dates2.weekday()==5 or dates2.weekday()==6):
            xlist2.append(dates2.strftime('%Y-%m-%d'))
            xlist2.remove(dates2.strftime('%Y-%m-%d'))
            # print(dates2)
            # print("removed")

        else:
            dates2=dates2.strftime('%Y-%m-%d')
            xlist2.append(dates2)
            holidaycount=0
            while (holidaycount<len(holidaylist)):

                if (dates2==holidaylist[holidaycount]):
                    xlist2.remove(dates2)
                    # print("removed" +dates2)
                holidaycount=holidaycount+1

        counter=counter+1
        # print(counter)
        # print(xlist2)



    xlist.extend(xlist2)






    new_list.extend(allvalueslist)
    # plt.plot(xlist,new_list, color='blue')
    data_tuples = list(zip(xlist,new_list))
    df1=pd.DataFrame(data_tuples, columns=['Date','Value'])
    print(df1)
    print("data")
    #define cutoff date
    dateforgraph=datetime.date.today()

    cutoff =dateforgraph.strftime('%Y-%m-%d')
    #sort dataframe because unsorted dates will not plot properly
    df1 = df1.sort_values(["Date"])
    #plot the whole dataframe in yellow
    plt.plot(df1.Date, df1.Value, c = "b", label = "before {}".format(cutoff))
    #plot the conditional data on top in red
    plt.plot(df1[df1.Date >= cutoff].Date, df1[df1.Date >= cutoff].Value, c = "r", label = "after {}".format(cutoff))
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation = 45)
    plt.legend()
    # plt.show()



    from bokeh.plotting import figure, output_file, show


    new_list.extend(allvalueslist)
    # plt.plot(xlist,new_list, color='blue')
    data_tuples = list(zip(xlist,new_list))
    df1=pd.DataFrame(data_tuples, columns=['Date','Value'])
    print(df1)
    print("data")
    #define cutoff date
    dateforgraph=datetime.date.today()

    cutoff =dateforgraph.strftime('%Y-%m-%d')
    #sort dataframe because unsorted dates will not plot properly
    df1 = df1.sort_values(["Date"])
    #plot the whole dataframe in yellow
    plt.plot(df1.Date, df1.Value, c = "b", label = "before {}".format(cutoff))
    #plot the conditional data on top in red
    plt.plot(df1[df1.Date >= cutoff].Date, df1[df1.Date >= cutoff].Value, c = "r", label = "after {}".format(cutoff))
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation = 45)
    plt.legend()
    # plt.show()


    # from bokeh.models import DatetimeTickFormatter
    #
    #
    #
    # # add a line renderer
    # from datetime import datetime
    print(xlist)
    # x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    print(new_list)
    # date_time_str = ['8/10/20', '8/11/20']
    # date_time=[]
    # index=0
    # from bokeh.models import SingleIntervalTicker, LinearAxis
    # while(index<len(xlist)):
    #
    #     date_time_obj = datetime.strptime(xlist[index], '%Y-%m-%d')
    #     date_time.append(date_time_obj)
    #     index=index+1
    # print(date_time)



    from bokeh.plotting import figure
    from bokeh.io import show
    import plotly.express as px
    import plotly.graph_objects as go
    df = DataFrame(xlist[19:22],columns=['Dates'])
    print (df)
    df['values']=new_list[19:22]
    df1=DataFrame(xlist[21:26],columns=['Dates1'])
    df1['values1']=new_list[21:26]

    # print(df)
    fig=go.Figure()


    df2 = DataFrame(xlist[10:22],columns=['Dates'])
    # print (df)
    df2['values']=new_list[10:22]
    df3=DataFrame(xlist[21:37],columns=['Dates1'])
    df3['values1']=new_list[21:37]


    fig.add_scatter(x=df['Dates'], mode='lines', y=df['values'], name='current values')
    fig.add_scatter(x=df1['Dates1'], y=df1['values1'], mode='lines', name='forecast values')



#     fig.update_layout(
#     # height=800,
#     title_text='Weekly Forecast'
# )
    # fig.show()
    graph = fig.to_html(full_html=False, default_height=500, default_width=1000)
    context = {'graph': graph}



    fig1 = go.Figure()

    # fig1=px.line(df2, x='Dates', y='values', range_x=['2020-08-06', '2020-08-31'])
    fig1.add_scatter(x=df2['Dates'], y=df2['values'], mode='lines', name='current values')
    fig1.add_scatter(x=df3['Dates1'], y=df3['values1'], mode='lines', name='forecasted values')

    # fig1.show()


    graph1 = fig1.to_html(full_html=False, default_height=500, default_width=1000)
    context = {'graph1': graph1}




# Select country

        # print(request.POST)
        # is_private = request.POST.get('key1', False);
    # return render(request, "page3.html", {'active_page' : 'page3.html', 'div_figure' : html_fig})
    return render(request, "page3.html", {'active_page' : 'page3.html', 'graph' : graph, 'graph1': graph1})
def services(request):
    return render(request, 'services.html')
