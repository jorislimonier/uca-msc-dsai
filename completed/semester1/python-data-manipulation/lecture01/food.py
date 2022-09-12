# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import csv
from collections import Counter, defaultdict


# %%
food = list(csv.DictReader(open('data/food-inspections.csv')))


# %%
food[0]


# %%
len(food)


# %%
# unique values of the 'Results' attribute
{ row['Results'] for row in food }


# %%
fail = [ row for row in food if row['Results'] == 'Fail' ]


# %%
len(fail)


# %%
worst = Counter(row['DBA Name'] for row in fail)
worst.most_common(5)


# %%
fail = [ { **row, 'DBA Name': row['DBA Name'].replace("'", '').upper() }
          for row in fail ]


# %%
worst = Counter(row['DBA Name'] for row in fail)
worst.most_common(5)


# %%
bad = Counter(row['Address'] for row in fail)
bad.most_common(5)


# %%
by_year = defaultdict(Counter)
for row in fail:
    by_year[row['Inspection Date'][:4]][row['Address']] += 1


# %%
by_year["2015"].most_common(5)


# %%
by_year["2019"].most_common(5)


# %%
airport = [ row for row in fail if row['Address'].startswith('11601 W TOUHY')]


# %%
{row['Address'] for row in airport }


# %%
c = Counter(row['AKA Name'] for row in airport)
c.most_common(5)


# %%
violations = airport[1]["Violations"].split("|")
violations


# %%
[v[:v.find("- Comments")].strip() for v in violations]


# %%
all_violations = [row["Violations"].split("|") for row in airport]
c = Counter()
for violation in all_violations:
    for v in violations:
        c[v[:v.find("- Comments")].strip()] += 1
c


# %%
c.most_common(5)


