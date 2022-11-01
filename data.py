from bs4 import BeautifulSoup
import requests
from csv import writer

with open('housing.csv', 'a', encoding='utf8', newline='') as f:
    thewriter = writer(f)
    header = ['Địa chỉ','Quận','Huyện', 'Loại hình nhà ở', 'Giấy tờ pháp lý', 'Số tầng', 'Số phòng ngủ'
        , 'Diện tích', 'Dài', 'Rộng', 'Giá/m2']
    thewriter.writerow(header)
    listCsv = []
    for i in range(9867):
        print(str(i+1902))
        linkurlalonhadat = 'https://alonhadat.com.vn/can-ban-nha-ha-noi-t1/trang-'+str(i+1902)+'.htm'
        linkalonhadatpage = requests.get(linkurlalonhadat)
        soudalonhadat = BeautifulSoup(linkalonhadatpage.content, 'html.parser')
        finddivcontentitems=soudalonhadat.find('div',class_='content-items')
        hrefalonhadats=finddivcontentitems.find_all('div',class_='content-item')

        for hrefalonhadat in hrefalonhadats:
         testNone=hrefalonhadat.find('a',class_='vip')
         if(testNone==None):
             testNone = hrefalonhadat.find('a')
         url = 'https://alonhadat.com.vn'+testNone['href']
         print(url)
         page = requests.get(url)
         soud = BeautifulSoup(page.content, 'html.parser')
         classCssPrice = soud.find('span', class_='price')
         price = classCssPrice.find('span', class_='value').text
         classCssArea = soud.find('span', class_='square')
         area = classCssArea.find('span', class_='value').text
         classCssAddress = soud.find('div', class_='address')
         address = classCssAddress.find('span', class_='value').text
         classCssInfor = soud.find('div', class_='infor')
         informationBDSs = classCssInfor.find_all('td')
         addressListlen=['NaN','NaN']
         addressList=address.split(',')
         if(len(addressList)<2):
             for address in addressList:
              addressListlen.append(address)
             addressList=addressListlen
         saveCsv = [address,addressList[len(addressList)-3],addressList[len(addressList)-2] ,informationBDSs[13].text, informationBDSs[15].text
             , informationBDSs[21].text, informationBDSs[27].text, area
             , informationBDSs[25].text, informationBDSs[19].text,
                    price]
         thewriter.writerow(saveCsv)



