import pandas as pd
import numpy as np

# Define the file names and directory
inputDirs = ['2009', '2010', '2011', '2012', '2013', '2014',
             '2015', '2016', '2017', '2017_1', '2018', '2019']
fileNames = ['6시간강수량', '강수확률', '습도',
             '일최고기온', '일최저기온', '풍속', '풍향', '하늘상태']

fileNames_eng = ['rainfall', 'probability of precipitation', 'humidity', 'highest temperature', 'lowest temperature',
                 'wind speed', 'wind direction', 'sky state']

# [mergeForecastDataset]
# Read the csv files in the assets/input and merge all of features
# Merge the each forecast dataset to one csv file
def mergeForecastDataset():
    for fileName in fileNames:
        allYears = pd.DataFrame()

        for dir in inputDirs:
            try:
                # Read the forecast csv file from input directory
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
            # Extract timestamp and value columns
            df = df.loc[:,['timestamp', 'value']]
            # Concatenate the dataframes
            allYears = pd.concat([df, allYears])
        # Write the merged dataframe to csv file
        allYears.set_index('timestamp', inplace=True)    
        allYears.to_csv("assets/output/부림동_" + fileName + ".csv")

# [makeVisitor]
# Method for Make the visitor dataset
def makeVisitor():
    years = ['2008', '2009', '2010', '2011', '2012', '2013', '2014',
         '2015', '2016', '2017', '2018', '2019', '2020']

    years.reverse()
    visitor = pd.DataFrame()
    for year in years:
        try:
            # Read the visitor csv file from input directory
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
        # Extract date and visitor columns
        data.columns = ['date', 'visitor']
        # concatenate the dataframes
        visitor = pd.concat([data, visitor])
    # Write the merged dataframe to csv file
    data['visitor'] = data['visitor'].str.replace(',', '').astype(int)
    visitor.reset_index(inplace=True, drop=True)
    visitor.set_index('date', inplace=True)
    visitor.to_csv("assets/output/visitors.csv")

# [visitorProcess]
# Method for Make day of year data
def visitorProcess():
    df = pd.read_csv('./assets/output/visitors.csv')
    
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.dayofyear
    df.to_csv('./assets/output/visitors.csv', index=False)

# [makeAtmosphere]
# Method for make the atmosphere dataset
def makeAtmosphere():
    # Define the years in reverse order
    years = ['2008', '2009', '2010', '2011', '2012', '2013', '2014',
             '2015', '2016', '2017', '2018', '2019']
    years.reverse()
    
    # Create an empty DataFrame to store the atmosphere data
    atmosphere = pd.DataFrame()
    
    # Iterate over each year
    for year in years:
        try:
            # Read the CSV file for the current year
            if year == '2018' or year == '2019':
                data = pd.read_csv('assets/input/Atmosphere/Atmosphere_'+year+'.csv', encoding='cp949', usecols=range(1, 12))
            else:
                data = pd.read_csv('assets/input/Atmosphere/Atmosphere_'+year+'.csv', encoding='cp949')
            
            # Rename the columns
            data.columns = ['city', 'town', 'established', 'position', 'timestamp',
                            'sulfur_dioxide', 'carbon_monoxide', 'ozone', 'nitrogen_dioxide',
                            'fine_dust_pm10', 'fine_dust_pm2.5']
            
            # Create a DataFrame from the data
            df = pd.DataFrame(data)
            
            # Extract only data close to Seoul Grand Park
            df = df.loc[(df['city'] == 'Gwacheon') & (df['established'] != 1991)]
            
            # Remove unnecessary columns
            df.drop(['established', 'position', 'fine_dust_pm2.5'], axis=1, inplace=True)
            
            # Remove columns with all the same values
            df.drop(['city', 'town'], axis=1, inplace=True)
            
            # Split the timestamp column into date and hour columns
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

        # Concatenate the current year's data with the overall atmosphere data
        atmosphere = pd.concat([df, atmosphere])
    
    # Group the atmosphere data by date
    group = atmosphere.groupby('date')
    days = atmosphere['date'].unique()
    
    # Calculate statistics for each pollutant
    sulfur_dioxide = getGroupDescribe(group['sulfur_dioxide'], 'sulfur_dioxide')
    carbon_monoxide = getGroupDescribe(group['carbon_monoxide'], 'carbon_monoxide')
    ozone = getGroupDescribe(group['ozone'], 'ozone')
    nitrogen_dioxide = getGroupDescribe(group['nitrogen_dioxide'], 'nitrogen_dioxide')
    fine_dust_pm10 = getGroupDescribe(group['fine_dust_pm10'], 'fine_dust_pm10')
    
    # Create a result DataFrame with the calculated statistics
    result = pd.DataFrame({'date': days})
    result = pd.concat([result, sulfur_dioxide, carbon_monoxide, ozone, nitrogen_dioxide, fine_dust_pm10], axis=1)
    result.reset_index(inplace=True, drop=True)
    result.set_index('date', inplace=True)
    
    # Save the result to a CSV file
    result.to_csv("assets/output/atmosphere.csv", encoding='cp949')

def getGroupDescribe(group, name):
    # Calculate statistics for a given group
    min = group.min().values.ravel()
    max = group.max().values.ravel()
    mean = group.mean().values.ravel()
    median = group.median().values.ravel()
    
    # Create a DataFrame with the calculated statistics
    result = pd.DataFrame({
        name+'_min': min,
        name+'_max': max,
        name+'_mean': mean,
        name+'_median': median
    })
    
    return result

# Combine the forecast dataset to one csv file
# Read the all forecast datasets and merge them
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

# Combine the all output datasets
def datasetCombine():
    # Read the atmosphere dataset
    result = pd.read_csv('./assets/output/atmosphere.csv')
    
    # Read the forecast datasets and merge them
    for file in fileNames:
        f = pd.read_csv('./assets/output/부림동_'+file+'.csv')
        f.rename(columns={'timestamp' : 'date'}, inplace=True)
        
        
        result = pd.merge(result, f, how='outer', on='date')
    
    # Read the day of week and merge it
    week = pd.read_csv('./assets/input/WeekSet.csv')
    result = pd.merge(result, week, how='outer', on='date')
    # Read the visitor dataset and merge it
    target = pd.read_csv('./assets/output/visitors.csv')
    result = pd.merge(result, target, how='right', on='date')
    
    print(result)
    result.set_index('date', inplace=True)    
    result.to_csv("assets/output/datasetCombine.csv")

    
def fillDirtyData():
    from sklearn.impute import KNNImputer
    
    df = pd.read_csv('./assets/output/datasetCombine.csv')

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
    df_filled['visitor'] = df_filled['visitor'].str.replace(',', '').astype(int)
    
    # KNN - 사용 안할 시 최종 결과 Dataframe 이름(하단 2줄) df_filled로 수정해야 함
    imputer = KNNImputer(n_neighbors=5) 
    df_imputed = pd.DataFrame(imputer.fit_transform(df_filled.iloc[:, 1:]), columns=df_filled.columns[1:])
    df_imputed.insert(0, 'date', df_filled['date'])
    
    df_filled.set_index('date', inplace=True)
    df_filled.to_csv("assets/output/dirtydataResult.csv")
    
def discretizeData():
    df = pd.read_csv('./assets/output/outlierResult.csv')
    # Rounding contents to integer values
    # Due to previous processing, datas have become float data
    df['sky state_min'] = df['sky state_min'].round()
    df['sky state_max'] = df['sky state_max'].round()
    df['sky state_mean'] = df['sky state_mean'].round()
    df['sky state_median'] = df['sky state_median'].round()
    # # Divide the value of the angle into 16 directions
    df['wind direction_min'] = (df['wind direction_min']/22.5).round()
    df['wind direction_max'] = (df['wind direction_max']/22.5).round()
    df['wind direction_mean'] = (df['wind direction_mean']/22.5).round()
    df['wind direction_median'] = (df['wind direction_median']/22.5).round()

    df.set_index('date', inplace=True)
    df.to_csv("assets/output/discretizeResult.csv")
    
def encodingData(scaler):
    from sklearn.preprocessing import OneHotEncoder
    df = pd.read_csv('./assets/output/discretizeResult.csv')
    
    # Feature creating
    # Encoding
    encoded_columns = [ 
                       'wind direction_min', 'wind direction_max', 
                       'wind direction_mean', 'wind direction_median', 
                        #   'sky state_min', 'sky state_max', 
                        #  'sky state_mean', 'sky state_median', 
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

    for column in df.columns:
        if column != 'date' and column not in encoded_columns and column != 'visitor':
        # if column != 'date' and column not in encoded_columns and column != 'visitor' and column != 'day':    
            df_encoded[column] = scaler.fit_transform(np.array(df_encoded[column]).reshape(-1, 1))

    df_encoded.to_csv("assets/output/encodingResult.csv")

# ========= Preprocessing methods =========

def find_outlier_z(data, featureName):
    threshold = 3

    mean = np.mean(data[featureName])
    std = np.std(data[featureName])

    z_score = [(y-mean)/std for y in data[featureName]]

    # masks = np.where(np.abs(z_score)>threshold)
    # print(masks)
    masks = data[np.abs(z_score) < threshold]

    return masks

def find_outlier_Turkey(data, featureName):
    # q1, q3 = np.percentile(data[featureName],[25,75])
    # q1 = np.percentile(data[featureName], 25)
    # q3 = np.percentile(data[featureName], 75)
    
    q1 = data[featureName].quantile(0.25)
    q3 = data[featureName].quantile(0.75)
    
    iqr = q3 - q1
    
    lower_bound = q1 - (iqr*1.5)
    upper_bound = q3 + (iqr*1.5)
    
    mask = data[(data[featureName] < upper_bound) & (data[featureName] > lower_bound)]

    return mask
    
# 연도별 outlier 제거
# df = pd.read_csv('./src/assets/output/부림동_6시간강수량.csv')
# print(detectOutlierAmongYear(df, 'rainfall_mean','2017-01-01'  ,'2018-01-01'))
def detectOutlierAmongYear(df, featureName, startYear, endYear):
    
    year_df = df[df['date'] >= startYear]
    year_df = year_df[year_df['date'] <= endYear]
    
    result = find_outlier_Turkey(year_df, featureName)
    
    return result


def deleteAllOutlier(df):
    for col in df.columns:
        if(col == 'date'):
            continue
        if(col == 'visitor'):
            continue
        df = find_outlier_z(df, col)
    df.to_csv('./assets/output/outlierResult.csv', index=False)
    
    
   #Feature Selection based on correlation
   # Correlation기반으로 feature selection
#    selected_feat = ['sulfur_dioxide_min', 'carbon_monoxide_max', 'ozone_max', 'nitrogen_dioxide_max', 'fine_dust_pm10_max', 'rainfall_mean',   'probability of precipitation_min', 'humidity_min', 'highest temperature_max','lowest temperature_min', 'wind speed_median', 'visitor']
def featureSelectionBasedOnCorrelation(selected_feat):
    df = pd.read_csv('./assets/output/finalDataset.csv')

    df_selected = df[selected_feat]

    target = df['visitor']
    week = df.filter(regex='week')
    date = df['date'] 
    df.drop(['date'], axis=1, inplace=True)

    # Feature Selection based on correlation
    corr_target = df_selected.corr(method='spearman').iloc[:,-1]
    corr_target = np.abs(corr_target)
    # corr_target.drop('visitor', inplace=True)

    print(corr_target)

    threshold = 0.005  # 상관 관계의 임계값 설정
    highly_correlated_features = []
    for i,col in enumerate(corr_target):
        if col > threshold:
            highly_correlated_features.append(i)

    selected_features = df_selected.iloc[:,highly_correlated_features]
    print(selected_features)
    result = pd.concat([date, selected_features,week, target], axis=1)
    result.to_csv('./assets/output/preprocessedDataset.csv', index=False)

    
# #Feature reduction based on features characteristics(PCA)
# #이거 사용하면 성능이 더 떨어집니다. 주석 해놓고 필요할때 수정해서 쓰세요
def featureReductionBasedOnCharacter():
    from sklearn.decomposition import PCA

    df = pd.read_csv('./assets/output/preprocessedDataset.csv')
    target = df['visitor']
    date = df['date']
    df.drop(['date'], axis=1, inplace=True)

    characters = ['sulfur', 'carbon', 'nitrogen', 'ozone',  'pm10','rainfall', 'highest temperature', 'lowest temperature','precipitation', 'humidity']
    char_cols = [df.loc[:,df.columns.str.contains(char)]for char in characters]


    pca_results = pd.DataFrame()
    for idx, char_col in enumerate(char_cols):
        pca = PCA(.95)
        pca_result = pca.fit_transform(char_col)
        pca_dataframe = pd.DataFrame(pca_result, columns=[characters[idx] +'_' + str(i) for i in range(pca_result.shape[1])])
        # pca_dataframe.set_index(pca_dataframe.columns[0], inplace=True)
        pca_results = pd.concat([pca_results, pca_dataframe], axis=1)

    print(pca_results)

    # for idx, pca_result in enumerate(pca_results):
    #     pca_df = pd.concat([pca_df, pca_result], axis=1)
    pca_df = pd.concat([date, pca_results,target], axis=1)

    pca_df.to_csv('./assets/output/pca.csv', index=False)
    
# # Feature Selection based on PCA
def doPCA():
    from sklearn.decomposition import PCA

    df = pd.read_csv('./assets/output/preprocessedDataset.csv')

    target = df['visitor']
    date = df['date'] 
    df.drop(['date', 'visitor'], axis=1, inplace=True)

    pca = PCA(.95)
    pca.fit(np.array(df, dtype=object))

    df_pca = pca.transform(df)
    df_pca = pd.DataFrame(df_pca)

    result = pd.concat([date, df_pca, target], axis=1)
    result.to_csv('./assets/output/preprocessedDataset.csv', index=False)
    
def dataSelectionByCondition(feature, threshold):
    df = pd.read_csv('./assets/output/preprocessedDataset.csv')

    df.drop(df[df[feature] > threshold].index, inplace=True)

    df.to_csv('./assets/output/preprocessedDataset.csv', index=False)