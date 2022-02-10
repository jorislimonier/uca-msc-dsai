# %%
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

# %%
driver = webdriver.Chrome(executable_path="/home/joris/chromedriver")


# %%
EASY_RDF_PATH = "https://www.easyrdf.org/converter"
driver.get(EASY_RDF_PATH)
# %%
text_box = driver.find_element(By.ID, "data")
text_box.click()
# %%
with open("lecture01.ttl") as f:
    ttl = f.read()
ttl
# %%
# for line in ttl.split("\n"):
line = ttl.split("\n")[4]
for char in ttl:
    # print(char)
    try:
        text_box.send_keys(r"{}".format(char))
    except Exception:
        print(char)
# %%
ttl.split("\n")[2]
text_box.send_keys(ttl)
# %%