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

def webpage3(request):
    us_holidays = holidays.US()
    holidaylist=[]


    # Print all the holidays in US in year 2018
    for ptr in holidays.US(years = 2020).items():
        #print(ptr[0])

        ptr = list(ptr)
        # print(ptr[0])
        print(ptr)
        # print(ptr[0])
        holidaylist.append(ptr[0].strftime('%Y-%m-%d'))
        print(holidaylist)
        print("holidaylist")
        #
        # [date_obj.strftime('%Y-%m%-d') for date_obj in holidaylist]
        # print(holidaylist[0])





        #holidaylist.append(ptr[0])
        #print(holidaylist)

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
        print(url)
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
                if(yesterday1 in us_holidays):
                    xlist.remove(yesterday1)
                # holidaycount=0
                # while (holidaycount<len(holidaylist)):
                #
                #     if (yesterday1==holidaylist[holidaycount]):
                #         print(holidaylist[holidaycount])
                #         xlist.remove(yesterday1)
                #         print("removed holiday" +yesterday1)
                #     holidaycount=holidaycount+1

            counter=counter-1
            print(counter)


        index=0
        while(index< len(xlist)):
            print(data['Time Series (Daily)'][xlist[index]]['4. close'])

            ylist.append(data['Time Series (Daily)'][xlist[index]]['4. close'])
            index=index+1
            print(index)



        new_list = []
        for item in ylist:
            new_list.append(float(item))

        print("newlist")

        print(new_list)

        df=DataFrame(new_list, columns=['stock_prices'])
    # print(df)
        values=df.values
        print(values)

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
        # counter=0
        # xlist2=[]
        # while(counter<21):
        #     dates2=datetime.date.today()+datetime.timedelta(days=counter)
        #     print(dates2)
        #     # print(dates2)
        #     # print("dates2")
        #     if(dates2.weekday()==5 or dates2.weekday()==6):
        #         xlist2.append(dates2.strftime('%Y-%m-%d'))
        #         xlist2.remove(dates2.strftime('%Y-%m-%d'))
        #         # print(dates2)
        #         # print("removed")
        #
        #     else:
        #         dates2=dates2.strftime('%Y-%m-%d')
        #         xlist2.append(dates2)
        #         holidaycount=0
        #         while (holidaycount<len(holidaylist)):
        #
        #             if(yesterday1 in us_holidays):
        #                 xlist.remove(yesterday1)
        #                 # print("removed" +dates2)
        #
        #
        #     counter=counter+1
        # predictions=[]
        #
        #
        #

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




        print(xlist[12:25])
        data_tuples = list(zip(xlist[9:36],new_list[9:]))
        df1=pd.DataFrame(data_tuples, columns=['Date','Value'])

        # html_table = df1.to_html(index=False)
        # return render('templates/home.html', {'html_table': html_table})
        print("data")
        #define cutoff date
        # dateforgraph=datetime.date.today()
        #
        # cutoff =dateforgraph.strftime('%Y-%m-%d')
        # #sort dataframe because unsorted dates will not plot properly
        # df1 = df1.sort_values(["Date"])
        # #plot the whole dataframe in yellow
        # plt.plot(df1.Date, df1.Value, c = "b", label = "before {}".format(cutoff))
        # #plot the conditional data on top in red
        # plt.plot(df1[df1.Date >= cutoff].Date, df1[df1.Date >= cutoff].Value, c = "y", label = "after {}".format(cutoff))
        # plt.xticks(rotation = 45)
        # plt.legend()
        # plt.gcf().autofmt_xdate()


        dateforgraph=datetime.date.today()

        cutoff =dateforgraph.strftime('%Y-%m-%d')
        #sort dataframe because unsorted dates will not plot properly
        df1 = df1.sort_values(["Date"])
        #plot the whole dataframe in yellow
        plt.plot(df1.Date, df1.Value, c = "b", label = "before {}".format(cutoff))



        #plot the conditional data on top in red


        # plt.plot(df1.Date,df1.Value,'bo-')
        #
        # # zip joins x and y coordinates in pairs
        # for x,y in zip(df1.Date,df1.Value):
        #
        #     label = "{:.2f}".format(y)
        #
        #     plt.annotate(label, # this is the text
        #              (x,y), # this is the point to label
        #              textcoords="offset points", # how to position the text
        #              xytext=(0,10), # distance from text to points (x,y)
        #              ha='center') # horizontal alignment can be left, right or center











        # x = np.sort(np.random.rand(15))
        # y = np.sort(np.random.rand(15))
        # names = np.array(list("ABCDEFGHIJKLMNO"))



        # fig.canvas.mpl_connect("motion_notify_event", hover)




        import mpld3
        from mpld3 import plugins



        # Define some CSS to control our custom labels
        css = """
        table
        {
          border-collapse: collapse;
        }
        th
        {
          color: #ffffff;
          background-color: ##98eade;
        }
        td
        {
          background-color: #cccccc;
        }
        table, th, td
        {
          font-family:Arial, Helvetica, sans-serif;
          border: 1px solid black;
          text-align: right;
        }
        """

        fig, ax = plt.subplots()
        fig. autofmt_xdate()
        # df = quandl.get("YALE/SPCOMP")
        # ts = df['S&P Composite']
        #
        labels = []
        i=9
        while (i<len(new_list[9:])):
            labels.append(str(new_list[i]))

            lines = plt.plot(df1.Date, df1.Value, marker='o', ls='-', ms=5, markerfacecolor='None',markeredgecolor='None',)

            ax.set_xticklabels(df1.Date, rotation = 100, ha="right")




            # plt.xticks(rotation = 45)

            tooltip = plugins.PointHTMLTooltip(lines[0], labels,
                                           voffset=10, hoffset=10, css=css)

            i=i+1
        plugins.connect(fig, tooltip)

        # mpld3.display()





        # plt.legend()
        # plt.show()



        # plt.show()



        # fig.autofmt_xdate(rotation=45)
        p=plt.plot(df1[df1.Date >= cutoff].Date, df1[df1.Date >= cutoff].Value, c = "y", label = "after {}".format(cutoff))




        fig = plt.gcf()
        # plt.close(fig)
        html_fig = mpld3.fig_to_html(fig,template_type='general')
    # return render(request, "index.html")





        # message="Hello {} I am learning".format(query)







# Select country

        # print(request.POST)
        # is_private = request.POST.get('key1', False);
        return render(request, "page3.html", {'active_page' : 'page3.html', 'div_figure' : html_fig})

def services(request):
    return render(request, 'services.html')
