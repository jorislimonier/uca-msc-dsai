# %%
import requests
from bs4 import BeautifulSoup
import urllib
# %%
BASE_URL = "https://catalog.ldc.upenn.edu/docs/LDC97S62/"
req = requests.get(url=BASE_URL)
soup = BeautifulSoup(req.content, "html.parser")
# %%
print(soup.prettify())

# %%
files = [tr.td.a.get("href") for tr in soup.tbody.find_all("tr")][1:]
files
# %%
urllib.request.urlretrieve(BASE_URL+files[1])
# %%
BASE_URL+files[1]