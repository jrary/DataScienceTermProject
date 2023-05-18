import pandas as pd
import numpy as np

inputDirs = ['2009', '2010', '2011', '2012', '2013', '2014',
             '2015', '2016', '2017', '2017_1', '2018', '2019']
fileNames = ['6시간강수량', '강수확률', '습도',
             '일최고기온', '일최저기온', '풍속', '풍향', '하늘상태']


def makeOutputDataset():
    for fileName in fileNames:
        allYears = pd.DataFrame()

        for dir in inputDirs:
            str = 'assets/input/' + dir+'/부림동_' + fileName + '_'+dir+'.csv'
            try:
                df = pd.read_csv('./src/assets/input/' + dir+'/부림동_' +
                                 fileName + '_'+dir+'.csv')
            except FileNotFoundError:
                print("Error: File not found!")
                continue
            except pd.errors.EmptyDataError:
                print("Error: File is empty")
                continue
            except Exception as e:
                print("Error:", str(e))
                continue
            print(df)
            allYears = pd.concat([df, allYears])

        allYears.to_csv("./src/assets/output/부림동_" + fileName + ".csv")


makeOutputDataset()
