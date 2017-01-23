library(tidyverse)
library(glmnet)
library(highcharter)
library(MASS)

train = read_csv("/Users/alexpapiu/Documents/Insight/Project/Data/sm_listings.csv")



train$price %>% density(30) %>% 
    hchart(xlim = c(0, 1000), color = "#67a9cf", area = T) %>% 
    hc_xAxis(title = list(text = "Daily Price in Dollars"),
             opposite = FALSE,
             plotLines = list(
                 list(color = "#ef8a62",
                      width = 2,
                      value = median(train$price)))) %>% 
    hc_yAxis(title = list(text = "Density"), 
             plotBands = list(
                 list(from = 25, to = JS("Infinity"),
             label = list(text = "This is a plotBand")))
             ) %>% 
    hc_add_theme(hc_theme_smpl())



#X = model.matrix(price ~ ., train)

model = lm(price ~ ., train)

train[3,]

predict(model, train) %>% hist(30)

hist(train$price, breaks = 30)

#rmse
mean((predict(model, train) - train$price)^2) %>%  sqrt()
train$price %>% log1p() %>%  hist(20)


out = boxcox(lm(train$price ~ 1))


plot(density((rpois(n = 10000, lambda = 4))))


rpois(n = 100000, lambda = 7) %>% hist() %>%  plot()



rnorm(1000, 10) %>% hist()
