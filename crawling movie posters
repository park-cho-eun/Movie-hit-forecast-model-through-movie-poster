from selenium import webdriver
import urllib.request
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

driver = webdriver.Chrome("C:\\Users\\clstm_\\OneDrive\\Desktop\\New Order\\chromedriver")
driver.implicitly_wait(3)
driver.get('https://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieList.do')

driver.find_element_by_id("cal_start").send_keys("2020-01-01")
driver.find_element_by_id("cal_end").send_keys(Keys.CONTROL + 'a');
driver.find_element_by_id("cal_end").send_keys(Keys.DELETE);
driver.find_element_by_id("cal_end").send_keys("2022-03-01")
driver.find_element_by_class_name("btn_more").click()
driver.find_element_by_xpath("/html/body/div[1]/div[2]/div[2]/div[3]/form/div[2]/div[8]/div/label[2]").click()
driver.find_element_by_xpath("/html/body/div[1]/div[2]/div[2]/div[3]/form/div[2]/div[8]/div/label[3]").click()

driver.find_element_by_xpath("/html/body/div[1]/div[2]/div[2]/div[3]/form/div[2]/div[6]/div/input").click()
element1 = driver.find_element_by_xpath("/html/body/div[3]/div[2]/div/table[1]/tbody/tr/td[1]/table[1]/tbody/tr[23]/th/input")
driver.execute_script("arguments[0].click();", element1)
driver.find_element_by_xpath("/html/body/div[3]/div[2]/div/div/span/a").click()

time.sleep(3)

driver.find_element_by_xpath("/html/body/div[1]/div[2]/div[2]/div[3]/form/div[2]/div[5]/div/input").click()
element2 = driver.find_element_by_xpath("/html/body/div[3]/div[2]/div/table[1]/tbody/tr/td[1]/table[1]/tbody/tr[1]/th/input")
driver.execute_script("arguments[0].click();", element2)
element3 = driver.find_element_by_xpath("/html/body/div[3]/div[2]/div/table[1]/tbody/tr/td[1]/table[1]/tbody/tr[2]/th/input")
driver.execute_script("arguments[0].click();", element3)
element4 = driver.find_element_by_xpath("/html/body/div[3]/div[2]/div/table[1]/tbody/tr/td[1]/table[1]/tbody/tr[3]/th/input")
driver.execute_script("arguments[0].click();", element4)
driver.find_element_by_xpath("/html/body/div[3]/div[2]/div/div/span/a").click()

time.sleep(5)

driver.find_element_by_class_name("btn_blue").click()

time.sleep(5)

driver.find_element_by_xpath("/html/body/div[1]/div[2]/div[2]/div[4]/form/div/a[1]").click()

time.sleep(3)

count = 691

for j in range(1, 11):
    driver.find_element_by_xpath("/html/body/div[1]/div[2]/div[2]/div[4]/form/div/ul/li[" + str(j) + "]/a").click()
    time.sleep(3)
    for i in range(1, 11):
        driver.find_element_by_xpath("/html/body/div[1]/div[2]/div[2]/div[4]/table/tbody/tr[" + str(i) + "]/td[1]/span/a").click()
        imgUrl = driver.find_element_by_css_selector('.fl.thumb').get_attribute("href")
        urllib.request.urlretrieve(imgUrl, "train" + str(count) + ".jpg")
        driver.find_element_by_class_name("close").click()
        count = count + 1

