import requests
from bs4 import BeautifulSoup
import pandas as pd

url=['https://www.reuters.com/markets/companies/HPQ.N','https://www.reuters.com/markets/companies/AAPL.OQ']
members=[]

for r in url:
    url_members=[]
    response=requests.get(r).text
    soup=BeautifulSoup(response,'html5lib')

    members_search=soup.find('div',{"class":"about-company-card__company-leadership__1mNWX"})
    try:
        for dt,dd in zip(members_search.find_all('dt'),members_search.find_all('dd')):
            url_members.append((dt.text.strip(),dd.text.strip()))

    except:pass
    members.append(url_members)

c=1
for member in members:
    df=pd.DataFrame(member,columns=['Name','jobTitle'])
    print(df)
    df.to_csv(str(c)+'.csv',index=False)
    c+=1
print("Data saved")
