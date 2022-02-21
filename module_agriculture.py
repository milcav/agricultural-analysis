import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn import preprocessing
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans

def loading_data():
    #function for loading data and correcting the date
    data = pd.read_excel('meteo_fert_soil_irrigation_new.xlsx', index_col = 0)
    data['date_of_ac'] = pd.to_datetime(data['date_of_ac'],
                                        dayfirst = False, yearfirst = False)
    data['Plantingda'] = pd.to_datetime(data['Plantingda'],
                                        dayfirst = False, yearfirst = False)
    #check if there are null vaules and drop them if their number is under 5%
    if data.isnull().any(axis=1).sum() < data.shape[0]*0.05:
        data = data.dropna()
    else:
        print('Too many null values')
    return data

def splitting_data(data):
    #fucntion for splitting data to multiple data frames
    weather_data = pd.concat([data.loc[: , 'rad1':'sm4'],
                              data.loc[: , 'yields']], axis=1)
    fertilizer_data = pd.concat([data.loc[: , 'AMOUNT':'microorganism'],
                                 data.loc[: , 'yields'],
                                 data.loc[: , 'Irrigation']], axis=1)
    soil_data = pd.concat([data.loc[: , 'AWCh1_sl5':'SLNI_60'],
                           data.loc[: , 'Soil_class'], 
                           data.loc[: , 'yields']], axis=1)
    geo_data =  pd.concat([data.loc[: , 'Latitude':'Longitude'],
                           data.loc[: , 'Soil_class'], 
                           data.loc[: , 'yields']], axis=1)
    time_data = pd.concat([data.loc[:, 'Plantingda'],
                           data.loc[: , 'date_of_ac'], 
                           data.loc[: , 'yields']], axis=1)
    #drawing_elements are used in plotting maps, taking only
    #soil types that have more then 100 samples
    s = geo_data['Soil_class'].value_counts()
    s = s[s > 100]
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(s)))
    drawing_elements = list(zip(s.index,s,colors))
    return weather_data, fertilizer_data, soil_data, geo_data, time_data, drawing_elements

def removing_outliners(df):
    #we will remove all rows that have outliner values in some columns
    df = df.copy()
    if 'Latitude' in df.columns:
    # this will apply only for geo_data that we use for plotting
        sd =df[df.columns.difference(['Latitude', 'Longitude', 'Soil_class'])]
        return df[(np.abs(stats.zscore(sd)) < 3).all(axis=1)]
    elif 'Soil_class' in df.columns:
    # in case we use 'Soil_class' in some other data that is not geo_data
        sd =df[df.columns.difference(['Latitude', 'Longitude', 'Soil_class'])]
        return df[(np.abs(stats.zscore(sd)) < 3).all(axis=1)]
    elif 'datetime64[ns]' in df.dtypes.values:
    #getting datetime data back in that shape after calculations
        df['date_of_ac'] = pd.to_numeric(df['date_of_ac'])
        df['Plantingda'] = pd.to_numeric(df['Plantingda'])
        df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
        df['date_of_ac'] = pd.to_datetime(df['date_of_ac'],
                                          dayfirst = False, yearfirst = False)
        df['Plantingda'] = pd.to_datetime(df['Plantingda'],
                                          dayfirst = False, yearfirst = False)
        return df
    return df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

def highest_spearman_corr_to_yield(df):
    #returning spearman correlation coeffficient higher then 0.38
    #in regards to yield with label of column
    df = df.copy()
    highest_corr = []
    for col in df.columns:
        if col == 'yields':
            continue
        if df[f'{col}'].dtype == 'object':
            df.loc[: , f'{col}'], unique = pd.factorize(df.loc[: , f'{col}'])
        elif df[f'{col}'].dtype == 'datetime64[ns]':
            df.loc[: , f'{col}'] = pd.to_numeric(df.loc[: , f'{col}'])
        corr_coef = df.loc[: , f'{col}'].corr(df.loc[: , 'yields'], method = 'spearman')
        if abs(corr_coef) > 0.38:
            highest_corr.append((corr_coef, col))
    highest_corr.sort(key = lambda x:x[0], reverse=True)
    return  highest_corr

def highest_mutual_info_to_yield(df):
    #returning mutual information coeffficient higher then 0.32
    #in regards to yield with label of column
    df = df.copy()
    highest_mi = []
    for col in df.columns:
        if col == 'yields':
            continue
        elif df[f'{col}'].dtype == 'datetime64[ns]':
            df.loc[: , f'{col}'] = pd.to_numeric(df.loc[: , f'{col}'])
        elif df[f'{col}'].dtype == 'object':
            le = preprocessing.LabelEncoder()
            df.loc[:,f'{col}'] = le.fit_transform(df.loc[: , f'{col}'])
        mi_coef = mutual_info_regression(df.loc[: , [f'{col}', 'yields']],
                                         df.loc[: , 'yields'])[0]
        if abs(mi_coef) > 0.32:
            highest_mi.append((mi_coef, col))
    highest_mi.sort(key = lambda x:x[0], reverse=True)
    return  highest_mi

def kde_plots(df,list_of_spear_coeff, list_of_mutual_info):
    #drawing kde of yield and feature with high spearman correlation
    #or mutual information correlation
    #if there are less then 3 unique values (encoded feature), draw a swarmplot
    for coeff, col in list_of_spear_coeff:
        if len(pd.unique(df[col])) < 3:
            plt.figure()
            sns.swarmplot(x=col, y='yields', data=df)
            plt.suptitle(f'Spearman coefficient is {coeff:.3f}')
            plt.savefig(f'KDE_spearman_{col}')
            plt.show()
            plt.close()
        else:
            plt.figure()
            sns.jointplot(x = col, y = 'yields', kind = 'kde', data = df)
            plt.suptitle(f'Spearman coefficient is {coeff:.3f}')
            plt.savefig(f'KDE_spearman_{col}')
            plt.show()
            plt.close()
    for coeff, col in list_of_mutual_info:
        if len(pd.unique(df[col])) < 3:
            plt.figure()
            sns.swarmplot(x=col, y='yields', data=df)
            plt.suptitle(f'Mutual information ceofficient is {coeff:.3f}')
            plt.savefig(f'KDE_mui_{col}')
            plt.show()
            plt.close()
        else:
            plt.figure()
            sns.jointplot(x =col, y = 'yields', kind = 'kde', data = df)
            plt.suptitle(f'Mutual information ceofficient is {coeff:.3f}')
            plt.savefig(f'KDE_mui_{col}')
            plt.show()
            plt.close()

def data_info(separated_data):
    #printing informations for different groups, data lost, number of features
    #and type of features, same as highest spearman and mutual information
    #coefficients and plots kde for
    filtered_df = removing_outliners(separated_data)
    data_loss = separated_data.shape[0]-filtered_df.shape[0]
    print(f'We had {separated_data.shape[0]} samples originaly.')
    if 'rad1' in separated_data.columns:
        print(f'For weather data we lost {data_loss} due to removing outliners'
              f'it has {separated_data.shape[1]-1} features and column for yield')
    elif 'AMOUNT' in separated_data.columns:
        print(f'For fertilizer data we lost {data_loss} due to removing outliners'
              f'it has {separated_data.shape[1]-1} features and column for yield')
    elif 'Soil_class' in separated_data.columns:
        print(f'For soil data we lost {data_loss} due to removing outliners'
              f'it has {separated_data.shape[1]-1} features and column for yield')
    highest_spear = highest_spearman_corr_to_yield(filtered_df)
    highest_mui = highest_mutual_info_to_yield(filtered_df)
    for coeff, name in highest_spear:
        print(f'For {name} spearman correlation is {coeff}')
    for coeff, name in highest_mui:
        print(f'For {name} mutual information is {coeff}')
    kde_plots(filtered_df, highest_spear, highest_mui)
    
def plotting_every_type_of_soil(geo_data, name, drawing_elements, size):
    #function for drawing sccatter plot on map, reciving only data with
    #coordinates, name of map, and drawing elements (geo_data SHOULD BE FILTERED)
    plt.figure(figsize=(17,17))
    #setting background with details
    stamen_terrain = cimgt.Stamen('terrain-background')
    graphic = plt.axes(projection=stamen_terrain.crs)
    graphic.add_feature(cfeature.BORDERS, linestyle=':')
    graphic.add_feature(cfeature.LAKES, alpha=0.5)
    graphic.add_feature(cfeature.RIVERS)
    # setting zoom on Mexico
    graphic.set_extent([-110, -84, 14, 27], ccrs.PlateCarree())         
    graphic.add_image(stamen_terrain, 8)
    for sclass, numb, color in drawing_elements:
        if size == 'yield':
            gd = geo_data.loc[geo_data['Soil_class'] == sclass]
            graphic.scatter(gd['Longitude'], gd['Latitude'], color=color,
                            s = gd['yields']*20 ,edgecolor='black',linewidths=0.5,
                            label = sclass, transform=ccrs.PlateCarree())
        else:
            gd = geo_data.loc[geo_data['Soil_class'] == sclass]
            graphic.scatter(gd['Longitude'], gd['Latitude'], color=color,
                            s = size ,edgecolor='black',linewidths=0.5,
                            label = sclass, transform=ccrs.PlateCarree())
    plt.legend()
    plt.title(f'{name}')
    plt.savefig(f'plot_map_{name}.png')
    plt.show()
    plt.close()
    
def plotting_map_for_each_soil(geo_data, drawing_elements):
    #ploting map for every soil_class that has more than 100 samples
    for sclass, numb, color in drawing_elements:
        gd = geo_data.loc[geo_data['Soil_class'] == sclass]
        plt.figure(figsize=(17,17))
        stamen_terrain = cimgt.Stamen('terrain-background')
        graphic = plt.axes(projection=stamen_terrain.crs)
        graphic.add_feature(cfeature.BORDERS, linestyle=':')
        graphic.add_feature(cfeature.LAKES, alpha=0.5)
        graphic.add_feature(cfeature.RIVERS)
        graphic.set_extent([-110, -84, 14, 27], ccrs.PlateCarree())         
        graphic.add_image(stamen_terrain, 8)
        graphic.scatter(gd['Longitude'], gd['Latitude'], color=color,
                        s =gd['yields']*20 ,edgecolor='black', linewidths=0.4, 
                        label = sclass, transform=ccrs.PlateCarree())
        plt.legend()
        mean_for_this_soil = gd[gd['Soil_class'] == sclass].mean()['yields']
        plt.title(f"Soil class {sclass} with mean of "
                  f"{mean_for_this_soil:.3f}")
        plt.savefig(f'Soil_class_{sclass}.png')
    plt.show()
    plt.close()

def plotting_map_for_different_sizes(geo_data, drawing_elements):
    # making 3 maps: highest 10% yield, lowest 10% yield, and full data
    top_df = geo_data.loc[geo_data['yields'] > geo_data['yields'].quantile(0.9)]
    low_df = geo_data.loc[geo_data['yields'] < geo_data['yields'].quantile(0.1)]
    data_frames = [(top_df, 'Highest 10%', 150), (low_df, 'Lowest 10%', 150),
                   (geo_data, 'Whole data', 'yield')]
    for data, name, size in data_frames:
        plotting_every_type_of_soil(data, name, drawing_elements, size)

def plots_for_soil_distribution(geo_data, drawing_elements):
    #plots for distribution of soil types
    types_soil_violin = [i[0] for i in drawing_elements[:3]]
    df_violin = geo_data.loc[geo_data['Soil_class'].isin(types_soil_violin)]
    types_soil_box = [i[0] for i in drawing_elements]
    df_box = geo_data.loc[geo_data['Soil_class'].isin(types_soil_box)]
    plt.figure()
    sns.violinplot(x='Soil_class', y='yields', data=df_violin,
                   whis = np.inf, order = types_soil_violin)
    plt.savefig('violin_for_soil.png', bbox_inches='tight')
    plt.show()
    plt.close()
    plt.figure()
    sns.set_style("whitegrid")
    sns.boxplot(y='Soil_class', x='yields', data=df_box,
                whis = np.inf, order = types_soil_box)
    plt.savefig('Boxplot_for_soil.png', bbox_inches='tight')
    plt.show()
    plt.close()
    plt.figure()
    sns.countplot(y='Soil_class', data=geo_data,
                  order = geo_data['Soil_class'].value_counts().index)
    plt.savefig('Countplot_for_soil.png', bbox_inches='tight')
    plt.show()
    plt.close()

def plotting_weather_type_map(weather_data, geo_data):
    #plot for clustering data based on weather type, inputs are RAW DATA WITH OUTLINERS
    #otherwise same as plotting_every_type_of_soil
    drawing_weather = pd.concat([weather_data,
                                 geo_data.loc[: , ['Latitude', 'Longitude']]], axis = 1)
    filtered_drawing_weather = removing_outliners(drawing_weather)
    kmeans =KMeans(n_clusters=4).fit(filtered_drawing_weather[filtered_drawing_weather.columns.difference(['Latitude', 'Longitude'])])
    colors = ['r', 'b', 'k', 'y']
    plt.figure(figsize=(17,17))
    stamen_terrain = cimgt.Stamen('terrain-background')
    graphic = plt.axes(projection=stamen_terrain.crs)
    graphic.add_feature(cfeature.BORDERS, linestyle=':')
    graphic.add_feature(cfeature.LAKES, alpha=0.5)
    graphic.add_feature(cfeature.RIVERS)
    graphic.set_extent([-110, -84, 14, 27], ccrs.PlateCarree())         
    graphic.add_image(stamen_terrain, 8)
    graphic.scatter(filtered_drawing_weather['Longitude'],
                    filtered_drawing_weather['Latitude'], s=150,
                    color=[colors[l_] for l_ in kmeans.labels_], label=kmeans.labels_,
                    transform=ccrs.PlateCarree(), alpha = .9, 
                    edgecolor = 'black', linewidths=0.4)
    plt.savefig('weather_cluster.png')
    plt.show()
    plt.close()
    
def info_about_geo_data(geo_data):
    #unusual connection between locations and yield
    fd =geo_data[geo_data.columns.difference(['Soil_class'])]
    geo_data = geo_data[(np.abs(stats.zscore(fd)) < 3).all(axis=1)]
    highest_spear = highest_spearman_corr_to_yield(geo_data)
    highest_mui = highest_mutual_info_to_yield(geo_data)
    sns.set()
    kde_plots(geo_data,highest_spear, highest_mui)
    #plotting unusal spot and Mexico City
    plt.figure(figsize=(17,17))
    stamen_terrain = cimgt.Stamen('terrain-background')
    graphic = plt.axes(projection=stamen_terrain.crs)
    graphic.add_feature(cfeature.BORDERS, linestyle=':')
    graphic.add_feature(cfeature.LAKES, alpha=0.5)
    graphic.add_feature(cfeature.RIVERS)
    graphic.set_extent([-110, -84, 14, 27], ccrs.PlateCarree())         
    graphic.add_image(stamen_terrain, 8)
    graphic.scatter(-99.13, 19.43, s=250, color='red', label='Mexico City',
                    transform=ccrs.PlateCarree(), alpha = 1)
    graphic.scatter(-101, 21.3, s=250, color='black', label='Unusual spot',
                    transform=ccrs.PlateCarree(), alpha = 1)
    plt.legend()
    plt.savefig('unusual_location.png')
    plt.show()
    plt.close()
