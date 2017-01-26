#maps:
library(tidyverse)
library(highcharter)
library(leaflet)
library(jsonlite)
library(rgdal)
library(tigris)
library(data.table)

train = read_csv("/Users/alexpapiu/Documents/Insight/Project/Data/clean_listings.csv")


train %>% 
    group_by(neighbourhood_cleansed) %>% 
    summarize(price = median(price), n = n()) %>% 
    filter(n > 15) -> price_by_nbd

#circle maps - one per listing:
factpal <- colorFactor(topo.colors(25), train$neighbourhood_cleansed)

pal <- colorNumeric(palette = "RdBu", domain = c(0, 300))
leaflet(data = train %>% sample_n(10000)) %>%  addProviderTiles("CartoDB.Positron") %>%
    addCircleMarkers(~longitude, ~latitude,
                stroke = FALSE, fillOpacity = 0.8, radius = 2,
                     color = ~pal(price))

#doing it by neighborhood:
geojson = readOGR("/Users/alexpapiu/Documents/Insight/Project/Data/neighbourhoods.geojson")

pal <- colorNumeric(na.color = NA,
    palette = "RdBu",
    domain = c(0, 200)
)



merge = geo_join(geojson, price_by_nbd, by_sp = "neighbourhood", by_df = "neighbourhood_cleansed")

leaflet() %>% addProviderTiles("CartoDB.Positron") %>% 
    addPolygons(data = merge, fillColor = ~pal(price), fillOpacity = 0.7,
                color = "white", weight = 1.5,
                popup = paste0(merge$neighbourhood_cleansed, "<br>", merge$price))
    


