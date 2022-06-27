# Let's install and load the required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(modeltime, modeltime.ensemble, modeltime.gluonts, ggthemes, tidymodels, tidyverse, 
               timetk, eurostat, lubridate, glmnet)

interactive <- FALSE
interactive <- TRUE
NAMQ_10_GDP <- get_eurostat("namq_10_gdp")

NAMQ_10_GDP <- NAMQ_10_GDP %>% filter(
  unit=="CLV10_MNAC",
  s_adj=="SCA",
  na_item %in% c("B1GQ","B1G","P5G","p51G","P6","P61","P62","P7","P71","P72","D1","D11","D12","B2A3G"),
  geo %in% c("EE", "LV","LT")
)

write.csv(NAMQ_10_GDP, file = "NAMQ_10_GDP.csv")
NAMQ_10_GDP <- read.csv("NAMQ_10_GDP.csv")
NAMQ_10_GDP <- NAMQ_10_GDP[c(2:7)]

library(DataExplorer)
DataExplorer::introduce(NAMQ_10_GDP)

# Chain linked volumes 2010=100
# Seasonally and callendar adjusted
# NA items:
# B1GQ - BVP
# B1G - Pridėtinė vertė
# P5G - Grynojo kapitalo formavimasis
# P51G - Grynojo ilgalaikio kapitalo formavimasis
# P6 - Prekių ir paslaugų eksportas 
# P61 - Prekių eksportas
# P62 - Paslaugų eksportas
# P7 - Prekių ir paslaugų importas
# P71 - Prekių importas
# P72 - Paslaugų importas
# D1 - Darbuotojų atlygis
# D11 - Darbo užmokestis
# D12 - Darbdavio mokamos socialinio draudimo įmokos
# B2A3G - Grynosios mišrios pajamos ir 
list_of_items <- c("B1GQ","B1G","P5G","p51G","P6","P61","P62","P7","P71","P72","D1","D11","D12","B2A3G")

esNAMQ_10_GDP <- spread(NAMQ_10_GDP, geo, values)
names(esNAMQ_10_GDP) <- c("unit", "s_adj", "id", "date", "EE", "LT", "LV")

lt_data <- dplyr::select(esNAMQ_10_GDP, c("id", "date", "LT")) %>% 
  filter(id == "B1GQ") # Lithuanian GDP data (B1GQ)

lt_data$date <- as.Date(lt_data$date)
class(lt_data$LT)
class(lt_data$id)

# DataExplorer::create_report(lt_data) # Some aoutomated EDA for the underlying data. It creates a HTML report.

lt_data %>%
  plot_time_series(date, LT, .interactive = interactive) + theme_clean() +
  ggtitle("Lithuanian GDP", "Eurostat data")
#theme_hc(bgcolor = "darkunica") +
#scale_colour_hc("darkunica")#theme_solarized(light = FALSE) #theme_economist()


# Split Data 80/20
splits <- initial_time_split(lt_data, prop = 0.9)

# Model 1: auto_arima ----
model_fit_arima_no_boost <- arima_reg() %>%
  set_engine(engine = "auto_arima") %>%
  fit(LT ~ date, data = training(splits))

summary(model_fit_arima_no_boost)

model_fit_arima_boosted <- arima_boost(
  min_n = 2,
  learn_rate = 0.015
) %>%
  set_engine(engine = "auto_arima_xgboost") %>%
  fit(LT ~ date + as.numeric(date) + factor(month(date, label = TRUE), ordered = F),
      data = training(splits))

summary(model_fit_arima_boosted)

# Model 3: ets ----
model_fit_ets <- exp_smoothing() %>%
  set_engine(engine = "ets") %>%
  fit(LT ~ date, data = training(splits))

model_fit_ets

# Model 4: prophet ----
model_fit_prophet <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(LT ~ date, data = training(splits))

model_fit_prophet

# Model 5: lm ----
model_fit_lm <- linear_reg() %>%
  set_engine("lm") %>%
  fit(LT ~ as.numeric(date) + factor(month(date, label = TRUE), ordered = FALSE),
      data = training(splits))

model_fit_lm

# Model 6: earth ----
model_spec_mars <- mars(mode = "regression") %>%
  set_engine("earth") 

recipe_spec <- recipe(LT ~ date, data = training(splits)) %>%
  step_date(date, features = "month", ordinal = FALSE) %>%
  step_mutate(date_num = as.numeric(date)) %>%
  step_normalize(date_num) %>%
  step_rm(date)

wflw_fit_mars <- workflow() %>%
  add_recipe(recipe_spec) %>%
  add_model(model_spec_mars) %>%
  fit(training(splits))

# Compile models into a Modeltime table
models_tbl <- modeltime_table(
  model_fit_arima_no_boost,
  model_fit_arima_boosted,
  model_fit_ets,
  model_fit_prophet,
  model_fit_lm,
  wflw_fit_mars
)

models_tbl

calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl

# Let's see how our models would have predicted the last year's Lithuanian economy
calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = lt_data
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = interactive
  ) + ggthemes::theme_clean() + ggtitle("Lithuania GDP: Actual vs Forecasts", "Eurostat data")

# Let's see which models fit the data best
calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = interactive
  )
# From the table we can see that 


# Let's refit out models to the full actual data
refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = lt_data)

# Let's see what the models predict for the Lithuanian GDP looking 3 years into the future
refit_tbl %>%
  modeltime_forecast(h = "3 years", actual_data = lt_data) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = interactive
  )

###
# Let's make a model ensemble to forecast the dynamic of the Lithuanian economy

splits <- time_series_split(lt_data, assess = "2 years", cumulative = TRUE)

splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, LT, .interactive = interactive)

# Let's create a feature engineering recipe that we'll use later on for meta-learning with Elastic Net
recipe_spec <- recipe(LT ~ date, training(splits)) %>%
  step_timeseries_signature(date) %>%
  step_rm(matches("(.iso$)|(.xts$)")) %>%
  step_normalize(matches("(index.num$)|(_year$)")) %>%
  step_dummy(all_nominal()) %>%
  step_fourier(date, K = 1, period = 12)

recipe_spec %>% prep() %>% juice()

# Train auto-arima
model_spec_arima <- arima_reg() %>%
  set_engine("auto_arima")

wflw_fit_arima <- workflow() %>%
  add_model(model_spec_arima) %>%
  add_recipe(recipe_spec %>% step_rm(all_predictors(), -date)) %>%
  fit(training(splits))

# Train the Facebook's Prophet model
model_spec_prophet <- prophet_reg() %>%
  set_engine("prophet")

wflw_fit_prophet <- workflow() %>%
  add_model(model_spec_prophet) %>%
  add_recipe(recipe_spec %>% step_rm(all_predictors(), -date)) %>%
  fit(training(splits))

# Train the Elastic Net Model
model_spec_glmnet <- linear_reg(
  mixture = 0.9,
  penalty = 4.36e-6
) %>%
  set_engine("glmnet")

# install.packages("glmnet")
wflw_fit_glmnet <- workflow() %>%
  add_model(model_spec_glmnet) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(splits))


# Let's once again create a Modeltime table to put our models in
lt_data_models <- modeltime_table(
  wflw_fit_arima,
  wflw_fit_prophet,
  wflw_fit_glmnet
)

lt_data_models

# Let's create an esemble of models
ensemble_fit <- lt_data_models %>%
  ensemble_average(type = "mean") # use the average
ensemble_fit <- lt_data_models %>%
  ensemble_average(type = "median") # use the median

ensemble_fit

# Now it's time for the forecast on test data
# Calibration
calibration_tbl <- modeltime_table(
  ensemble_fit
) %>%
  modeltime_calibrate(testing(splits)) # specify the splits

# Forecast vs Test Set
calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = lt_data
  ) %>%
  plot_modeltime_forecast(.interactive = interactive) + theme_clean() + 
  ggtitle("Lithuania GDP: Actual vs Ensemble Forecast", "Eurostat data, my calculations")

# Refit on Full Data and Forecast the Future

refit_tbl <- calibration_tbl %>%
  modeltime_refit(lt_data)

refit_tbl %>%
  modeltime_forecast(
    h = "2 years",
    actual_data = lt_data
  ) %>%
  plot_modeltime_forecast(.interactive = interactive) + theme_clean() + 
  ggtitle("Lithuanian GDP: Forecast 2 years into the future using combined model ensemble", 
          "Eurostat data, my calculations using modeltime.ensemble framework, using median forecasts from 3 models (Auto-Arima, Prophet and glmnet (Elastic Net Model))")


##
# gluon-ts
# 
install_gluonts()

data <- m4_hourly %>%
  select(id, date, value) %>%
  group_by(id) %>%
  mutate(value = standardize_vec(value)) %>%
  ungroup()

data
