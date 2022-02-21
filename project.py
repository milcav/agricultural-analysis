"""
Created on Thu Apr  30 23:00:35 2020

author: Miljan Cavic
"""

import module_agriculture as ma

# =============================================================================
# First we load and split data
# =============================================================================
data = ma.loading_data()
(weather_data, fertilizer_data, soil_data,
 geo_data, time_data, drawing_elements) = ma.splitting_data(data)
elementary_data_sets = [weather_data, fertilizer_data, soil_data]

# =============================================================================
# First we run usual informations and plots for elementary data sets: wheater,
# fertilizing, soil
# =============================================================================
for data_set in elementary_data_sets:
    ma.data_info(data_set)
    
# =============================================================================
# Now we take geographic data and we draw some maps with locations and types of
# soil, we took only soil types that have more than 100 sampels
# =============================================================================
filtered_geo_data = ma.removing_outliners(geo_data)
ma.plotting_map_for_each_soil(filtered_geo_data, drawing_elements)
ma.plotting_map_for_different_sizes(filtered_geo_data, drawing_elements)

# =============================================================================
# Showing distribution of some soil types
# =============================================================================
ma.plots_for_soil_distribution(geo_data, drawing_elements)

# =============================================================================
# On weather data we can nearly conclude what type of climate is at that point
# =============================================================================
ma.plotting_weather_type_map(weather_data, geo_data)

# =============================================================================
# While testing code i run at some unusual connection between coordinates and
# yield that i draw on map
# =============================================================================
ma.info_about_geo_data(geo_data)