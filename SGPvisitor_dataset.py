import numpy as np
import pandas as pd

years = ['2008', '2009', '2010', '2011', '2012', '2013', '2014',
         '2015', '2016', '2017', '2018', '2019']
years.reverse()

allYears = pd.DataFrame()
for year in years:
    try:
        data = pd.read_excel(
            './src/assets/SGPyear/Visitor_'+year+'.xlsx', skiprows=2, usecols=[0, 5])
    except FileNotFoundError:
        print("Error: File not found!")
        continue
    except pd.errors.EmptyDataError:
        print("Error: File is empty")
        continue
    except Exception as e:
        print("Error:", str(e))
        continue
    df = pd.DataFrame(data)
    df.columns = ['date', 'visitor']
    df['date'] = df['date'].str.split(" ", 1, expand=True)
    allYears = pd.concat([df, allYears])

allYears.to_csv("./src/assets/output/visitors.csv")
