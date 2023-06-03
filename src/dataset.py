import pandas as pd
import numpy as np

# Define the file names and directory
inputDirs = ['2009', '2010', '2011', '2012', '2013', '2014',
             '2015', '2016', '2017', '2017_1', '2018', '2019']
fileNames = ['6시간강수량', '강수확률', '습도',
             '일최고기온', '일최저기온', '풍속', '풍향', '하늘상태']
fileNames_eng = ['rainfall', 'probability of precipitation', 'humidity', 'highest temperature', 'lowest temperature',
                 'wind speed', 'wind direction', 'sky state']

# Read the csv files in the assets/input and merge all of features


def mergeForecastDataset():
    for fileName in fileNames:
        allYears = pd.DataFrame()

        for dir in inputDirs:
            try:
                df = pd.read_csv('./assets/input/' + dir+'/부림동_' +
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
            df = df.loc[:,['timestamp', 'value']]
            allYears = pd.concat([df, allYears])
        allYears.set_index('timestamp', inplace=True)    
        allYears.to_csv("assets/output/부림동_" + fileName + ".csv")


# Visitor
def makeVisitor():
    years = ['2008', '2009', '2010', '2011', '2012', '2013', '2014',
         '2015', '2016', '2017', '2018', '2019', '2020']

    years.reverse()
    visitor = pd.DataFrame()
    for year in years:
        try:
            data = pd.read_csv(
                'assets/input/SGPyear/Visitor_'+year+'.csv', skiprows=2, usecols=[0, 5])
        except FileNotFoundError:
            print("Error: File not found!")
            continue
        except pd.errors.EmptyDataError:
            print("Error: File is empty")
            continue
        except Exception as e:
            print("Error:", str(e))
            continue
        data.columns = ['date', 'visitor']
        visitor = pd.concat([data, visitor])
    data['visitor'] = data['visitor'].str.replace(',', '').astype(int)
    visitor.reset_index(inplace=True, drop=True)
    visitor.set_index('date', inplace=True)
    visitor.to_csv("assets/output/visitors.csv")


def makeAtmosphere():
    years = ['2008' , '2009', '2010', '2011', '2012', '2013', '2014',
         '2015', '2016', '2017', '2018', '2019' ]
    years.reverse()
    atmosphere = pd.DataFrame()
    for year in years:
        try:
            if year == '2018' or year == '2019':
                data = pd.read_csv('assets/input/Atmosphere/Atmosphere_'+year+'.csv', encoding='cp949', usecols=range(1,12))
            else:
                data = pd.read_csv('assets/input/Atmosphere/Atmosphere_'+year+'.csv', encoding='cp949')
            data.columns = ['city', 'town', 'established', 'position', 'timestamp', 
                                'sulfur_dioxide', 'carbon_monoxide', 'ozone', 'nitrogen_dioxide', 
                                'fine_dust_pm10', 'fine_dust_pm2.5']
            df = pd.DataFrame(data)
            # Extract only data close to Seoul Grand Park
            df = df.loc[(df['city']=='Gwacheon') & (df['established']!=1991)]
            # Unnecessary data: "설치년도", "측정망 정보"   
            # Unavailable data (many missing data): "미세먼지PM2.5농도값(μg/m³)"
            df.drop(['established', 'position', 'fine_dust_pm2.5'], axis=1, inplace=True)
            # Unnecessary data (because all data values are same): "시군명", "측정소명"
            df.drop(['city', 'town'], axis=1, inplace=True)
            # Split the data with Date/Time
            df[['date', 'hour']] = df['timestamp'].str.split(" ", 1, expand=True)
            df.drop(['timestamp'], axis=1, inplace=True)
        except FileNotFoundError:
            print("Error: File not found!")
            continue
        except pd.errors.EmptyDataError:
            print("Error: File is empty")
            continue
        except Exception as e:
            print("Error:", str(e))
            continue

        atmosphere = pd.concat([df, atmosphere])
    
    group = atmosphere.groupby('date')
    days = atmosphere['date'].unique()
    
    sulfur_dioxide = getGroupDescribe(group['sulfur_dioxide'], 'sulfur_dioxide')
    carbon_monoxide = getGroupDescribe(group['carbon_monoxide'], 'carbon_monoxide')
    ozone = getGroupDescribe(group['ozone'], 'ozone')
    nitrogen_dioxide = getGroupDescribe(group['nitrogen_dioxide'], 'nitrogen_dioxide')
    fine_dust_pm10 = getGroupDescribe(group['fine_dust_pm10'], 'fine_dust_pm10')
    
    result = pd.DataFrame({'date' : days})
    result = pd.concat([result, sulfur_dioxide,carbon_monoxide,ozone,nitrogen_dioxide,fine_dust_pm10], axis=1)
    result.reset_index(inplace=True, drop=True)
    result.set_index('date', inplace=True)
    result.to_csv("assets/output/atmosphere.csv", encoding='cp949')

def getGroupDescribe(group, name):
    min = group.min().values.ravel()
    max = group.max().values.ravel()
    mean = group.mean().values.ravel()
    median = group.median().values.ravel()
    # std = group.std().values.ravel()
    
    result = pd.DataFrame({
            name+'_min': min,
            name+'_max': max,
            name+'_mean': mean,
            name+'_median': median,
            # name+'_std': std
        })
    
    return result

def addOutputDataset():
    for i, file in enumerate(fileNames):
        df = pd.read_csv('./assets/output/부림동_'+file+'.csv')
        
        days = df['timestamp'].unique()
        
        name = fileNames_eng[i]
        group = getGroupDescribe(group=df.groupby('timestamp'), name=name)
        
        
        temp_df = pd.DataFrame({
            'timestamp': days,
        })
        
        temp_df = pd.concat([temp_df, group], axis=1)
        
        temp_df.set_index('timestamp', inplace=True)    
        temp_df.to_csv("assets/output/부림동_" + file + ".csv")

def makeFinalResult():
    result = pd.read_csv('./assets/output/atmosphere.csv')
    
    for file in fileNames:
        f = pd.read_csv('./assets/output/부림동_'+file+'.csv')
        f.rename(columns={'timestamp' : 'date'}, inplace=True)
        
        
        result = pd.merge(result, f, how='outer', on='date')
    
    week = pd.read_csv('./assets/input/WeekSet.csv')
    result = pd.merge(result, week, how='outer', on='date')
    target = pd.read_csv('./assets/output/visitors.csv')
    # target = target.drop(target[target['date'] <= '2017-12-31'].index)
    result = pd.merge(result, target, how='right', on='date')
    
    print(result)
    result.set_index('date', inplace=True)    
    result.to_csv("assets/output/finalDataset.csv")
    
def fillDirtyData():
    from sklearn.impute import KNNImputer
    
    df = pd.read_csv('./assets/output/finalDataset.csv')

    # Fill the NaN value with average value of the same date in the previous year 
    df['date_md'] = pd.to_datetime(df['date']).dt.strftime('%m-%d')
    df_filled = df.copy()
    
    ## 여기서부터 바꿔가면서 진행
    # NaN값 drop
    df_filled.dropna(thresh = 2)
    # 이전 값으로 채우기
    # df_filled.fillna(method='bfill')
    # 평균으로 채우기
    # for column in df.columns:
    #     if column != 'date' and column != 'date_md':
    #         if df[column].isnull().any():
    #             df_filled[column].fillna(df.groupby('date_md')[column].transform('mean'), inplace=True)
    
    # 날짜별로 합쳤던 column 삭제하는 부분 - 지우면 안됨
    df_filled.drop('date_md', axis=1, inplace=True)
    
    # KNN - 사용 안할 시 최종 결과 Dataframe 이름(하단 2줄) df_filled로 수정해야 함
    df_filled['visitor'] = df_filled['visitor'].str.replace(',', '').astype(int)
    imputer = KNNImputer(n_neighbors=5)  # Choose the number of neighbors according to your dataset
    df_imputed = pd.DataFrame(imputer.fit_transform(df_filled.iloc[:, 1:]), columns=df_filled.columns[1:])
    df_imputed.insert(0, 'date', df_filled['date'])
    
    df_imputed.set_index('date', inplace=True)
    df_imputed.to_csv("assets/output/preprocessedDataset.csv")
    
def discretizeData():
    df = pd.read_csv('./assets/output/preprocessedDataset.csv')
    # Rounding contents to integer values
    # Due to previous processing, datas have become float data
    df['sky state_min'] = df['sky state_min'].round()
    df['sky state_max'] = df['sky state_max'].round()
    df['sky state_mean'] = df['sky state_mean'].round()
    df['sky state_median'] = df['sky state_median'].round()
    # Divide the value of the angle into 16 directions
    df['wind direction_min'] = (df['wind direction_min']/22.5).round()
    df['wind direction_max'] = (df['wind direction_max']/22.5).round()
    df['wind direction_mean'] = (df['wind direction_mean']/22.5).round()
    df['wind direction_median'] = (df['wind direction_median']/22.5).round()

    df.set_index('date', inplace=True)
    df.to_csv("assets/output/preprocessedDataset.csv")
    
def encodingData():
    from sklearn.preprocessing import OneHotEncoder
    df = pd.read_csv('./assets/output/preprocessedDataset.csv')
    
    # Feature creating
    # Encoding
    encoded_columns = [ #'wind direction_min', 'wind direction_max', 
                       'wind direction_mean', 'wind direction_median', 
                          'sky state_min', 'sky state_max', 
                         'sky state_mean', 'sky state_median', 
                            'weekday']
    encoder = OneHotEncoder()
    df_encoded = encoder.fit_transform(df[encoded_columns])

    # Convert encoded data to new dataFrame
    encoded = pd.DataFrame(df_encoded.toarray(), columns=encoder.get_feature_names_out(encoded_columns))

    # Combine encoded dataFrame with original dataFrame
    df_encoded = pd.concat([df.drop(columns=encoded_columns), encoded], axis=1)
    df_encoded.set_index('date', inplace=True)

    # Data normalization
    # standardScaler
    from sklearn.preprocessing import StandardScaler
    for column in df.columns:
        if column != 'date' and column not in encoded_columns and column != 'visitor':
            scaler = StandardScaler()
            df_encoded[column] = scaler.fit_transform(np.array(df_encoded[column]).reshape(-1, 1))

    df_encoded.to_csv("assets/output/preprocessedDataset.csv")

def find_outlier_z(data, featureName):
    threshold = 3

    mean = np.mean(data[featureName])
    std = np.std(data[featureName])

    z_score = [(y-mean)/std for y in data[featureName]]

    # masks = np.where(np.abs(z_score)>threshold)
    # print(masks)
    masks = data[np.abs(z_score) < threshold]

    return masks
    
# 연도별 outlier 제거
def detectOutlierAmongYear(df, featureName):
    
    year_df = df[df['timestamp'] < '2018-01-01']
    year_df = year_df[year_df['timestamp'] >= '2017-01-01']
    
    result = find_outlier_z(year_df, featureName)
    
    return result
    