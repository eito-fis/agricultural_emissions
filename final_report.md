Predicting Agricultural Emissions
================

  - [Background](#background)
  - [Question](#question)
  - [Dataset](#dataset)
  - [Modeling](#modeling)
      - [Data](#data)
      - [Model](#model)
      - [Validation](#validation)
  - [Conclusion](#conclusion)

## Background

Global warming is one of the primary global problems we face today. The
world is heating up at an unprecedented rate due to emissions produced
by human activity, and we are quickly reaching a point of no return. Of
all emissions produced globally, about 11% can be attributed to
agriculture (1). While not as impactful as something like
transportation, which makes up over 30% (1), cutting emissions in any
sector can make a difference.

Agricultural greenhouse gas emissions come from several sources; the
three primary ones are agricultural soil management, enteric
fermentation, and manure management:

  - Agricultural soil management: “Nitrous oxide is produced naturally
    in soils through the microbial processes of nitrification and
    de-nitrification.” (2)

  - Enteric fermentation: “Methane is produced as part of the normal
    digestive processes in animals. During digestion, microbes in the
    animal’s digestive system ferment feed. This process, called enteric
    fermentation, produces methane as a by-product which can be emitted
    by the exhaling and belching of the animal.” (2)

  - Manure management: “Methane is produced by the anaerobic (without
    oxygen) decomposition of manure. When manure is handled as a solid
    or deposited naturally on grassland, it decomposes aerobically (with
    oxygen) and creates little methane emissions. However, manure stored
    as a liquid or slurry in lagoons, ponds, tanks or pits, decomposes
    anaerobically and creates methane emissions.” (2)

## Question

**Can we predict a country’s agricultural emissions given the raw
agricultural and livestock production of the country that year?**

When doing background research, we noticed that the processes that
result in agricultural emissions are pretty direct. Fermentation of
Nitrogen sources in the agricultural soils leads to Nitrous oxide being
produced. This can be reframed as: the more crop farming we do, the more
Nitrous oxide release we get (as they are directly related). Enteric
fermentation is also directly related to the process of livestocks’
digestive systems. So, the more livestock we have, the more Methane we
have released into the environment as a result of enteric fermentation.
The same idea applies to manure management. If these processes are so
directly related to the release of emissions and global warming, we
should be able to predict agricultural emissions from data of
agricultural and livestock production.

## Dataset

We found the FAOStat database (3), which is curated by the Food and
Agriculture Organization of the United Nations (FAO). The FAOStat
database aggregates many smaller datasets on various specific metrics as
related to worldwide agriculture. Of these subsets, we selected the
datasets on crop production, live animals and agricultural emissions.

From the datasets we chose, we attain area harvested, yield, and
production quantity of 129 types of crops and the amount of stock of 14
types of livestock for 225 countries over years 1961 - 2017. The
agricultural emissions dataset also gives us access to the amount of CH4
and N2O emissions for these years and countries, each of which we
convert to terms of gigagrams of CO2 then combine. We link the different
datasets through year and country, with the goal of predicting the CO2
emissions using the crop and livestock data.

``` r
df_all_na <- read_csv("./data/processed/combined_total_na.csv")
```

For uncertainty, because our data is collected of our sample size (the
world) and is not representative of a larger sample size, then we can
say that there is no uncertainty because there is no sampling.

## Modeling

### Data

To start, we process this data so that each observation consists of the
raw agricultural output and CO2 emissions due to agriculture of a
country for a given year. NA values were replaced by 0s, as they tended
to indicate that a country didn’t produce any of the given produce that
year, and aggregate columns (like Goat and Sheep) were discarded in
favor of their individual components. We also identified a list of
`Area`s that grouped multiple countries and removed them. This included
China, as it is broken down into 4 major sections that otherwise exist
in the dataset. Finally, we removed rows where the CO2 emissions were 0,
as this means there were no recorded emissions.

``` r
# List of areas that (usually) aren't actually countries
areas_bad = c("World", "Americas", "Asia", "Australia and New Zealand", "China", "Central Asia", "Central America", "Eastern Africa", "Eastern Asia", "Eastern Euorpe", "Europe", "Land Locked Developing Countries", "Least Developed Countries", "Low Income Food Deficit Countries", "Middle Africa", "Net Food Importing Developing Countries", "Northern Europe", "Northern Africa", "Northern America", "Pacific Islands Trust Territory", "Southern Asia", "South America", "Southern Africa", "Small Island Developing States", "Serbia and Montenegro", "Western Asia", "Western Africa")
df_processed <-
  df_all_na %>%
  rename(CO2 = `Agriculture total|CO2`) %>%
  select_if(~ !is.logical(.)) %>%
  replace(is.na(.), 0) %>%
  select(-contains("and")) %>%
  filter(CO2 != 0) %>%
  filter(!(Area %in% areas_bad))
```

Next, we split the data into the train set and the hold-out test set.

``` r
set.seed(42)
emissions_split = initial_split(df_processed, prop = 0.8)
emissions_train = training(emissions_split)
emissions_test = testing(emissions_split)
```

Then, we define several pre-processing steps. First, we drop the `Year`
and `Area` columns as they are not inputs we want the model to use. We
also remove columns with zero variance then standardize the variables to
have a mean of 0 and a standard deviation of 1. Both steps are important
for fitting a lasso regression model.

``` r
emissions_rec <-
  recipe(`CO2` ~ ., data = emissions_train) %>%
  step_rm(Year, Area) %>%
  step_zv(all_numeric(), -all_outcomes()) %>%
  step_normalize(all_numeric(), -all_outcomes())

emissions_prep <-
  emissions_rec %>%
  prep()
```

### Model

For the modeling portion, we decided to use lasso regression. As we had
a large amount of inputs, we wanted to leverage lasso regression’s
property of pushing coefficients to 0 in order to have a more tractable
end model. To start, we define our model with a naively chosen penalty
of `0.1` then fit on the train set to get a sense of performance.

We visualize the coefficients of the model with the largest magnitude to
get a sense of what the model values in the prediction process. Seeing
cattle as the greatest factor is encouraging - we know from other
research that cattle greatly contribute to multiple types of emissions.

``` r
df_coeff %>% 
  mutate(
    term = fct_reorder(term, estimate)
  ) %>% 
  arrange(desc(abs(estimate))) %>% 
  filter(term != "(Intercept)") %>%
  head(20) %>%
  ggplot(aes(
      x = estimate,
      y = term
    )) +
  geom_col()
```

<img src="final_report_files/figure-gfm/unnamed-chunk-6-1.png" width="100%" />

We then sweep across a range of L1 penalty parameters on our train set
to determine the setting that minimizes root mean squared error on our
data.

### Validation

The chosen penalty is used to fit a model on our train set to create our
final model. We evaluate the final fit on our test set to see how the
model performs on data that it has not seen during training. Here, we
achieve a r-squared of about `0.0027`. Note that tidymodels seems to
have 1 as the best possible outcome for r-squared, so we manually
adjust. Given that this r-squared value is small, we believe that we
have a strong model for predicting agricultural emissions.

``` r
final_fit <- last_fit(
  final_lasso,
  emissions_split
)

final_fit %>%
  collect_metrics() %>%
  select(-.config, -.estimator) %>%
  filter(.metric == "rsq") %>%
  mutate(.estimate = 1 - .estimate) %>%
  rename("Metric" = .metric, "Estimate" = .estimate)
#> # A tibble: 1 x 2
#>   Metric Estimate
#>   <chr>     <dbl>
#> 1 rsq     0.00269
```

However we have not addressed the largest source of uncertainty in our
process - the random train / test split. To account for this, we apply
10-fold cross validation on our train data which allows us to quantify
how our model’s performance might vary based on the initial split. We
use the produced matrix to make a 95% confidence interval of about
`0.0024 - 0.0046` for r-squared. Given that the high end of the
r-squared interval is still low and the overall range of the interval is
relatively speaking small, we believe that the train / test split does
not significantly affect the performance of our model.

``` r
set.seed(44)
folds <- vfold_cv(emissions_train, v = 10)
confidence_level = 0.95

results <-
  fit_resamples(
    final_lasso,
    folds
  ) %>%
  collect_metrics()

stat <- results %>% filter(.metric == "rsq")
lo <- pull(stat, "mean") - qnorm( 1 - (1 - confidence_level) / 2 ) * pull(stat, "std_err")
hi <- pull(stat, "mean") + qnorm( 1 - (1 - confidence_level) / 2 ) * pull(stat, "std_err")
cat("95% Confidence Interval for r^2\n", "Lower bound:", 1 - hi, "\n", "Upper bound:", 1 - lo)
#> 95% Confidence Interval for r^2
#>  Lower bound: 0.002380999 
#>  Upper bound: 0.004641898
```

As a final validation step, we produce two graphs. The first is a graph
of the predicted CO2 versus the actual CO2 for all observations in our
test set. Here, we note that the majority of points lie around the line
y = x, which means that the models predictions are close to the actual
value for many of the observations in the test set.

``` r
predictions <-
  collect_predictions(
    last_fit(
      final_lasso,
      emissions_split,
    )
  ) %>%
  pull(".pred")

emissions_test %>%
  mutate("Predicted CO2" = predictions) %>%
  ggplot(aes(CO2, `Predicted CO2`)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "grey")
```

<img src="final_report_files/figure-gfm/unnamed-chunk-10-1.png" width="100%" />

The second is a graph of the actual CO2 for a set of countries over time
next to the predicted CO2 for the same set of countries over time. We
filter for countries with more than 10 data points in the test set and
exclude the many countries with a small amount of emissions to make a
clearer graph. This does potentially ignore issues that we might have in
predicting low emission countries, but considering we have validated our
results several ways we believe this is alright. Looking at the graph,
it is clear that the general shape of the emissions over time for both
predicted and actual CO2 are very similar.

``` r
emissions_test %>%
  mutate("Predicted CO2" = predictions) %>%
  filter(CO2 > 50000) %>%
  group_by(Area) %>%
  filter(n() > 10) %>%
  ungroup() %>%
  pivot_longer(
    cols = c("CO2", "Predicted CO2"),
    names_to = "metric",
    values_to = "value"
  ) %>%
  ggplot(aes(x = Year, group = Area, color = Area)) +
  geom_line(aes(y = value)) +
  theme(legend.position = "none") +
  facet_wrap(~ metric)
```

<img src="final_report_files/figure-gfm/unnamed-chunk-11-1.png" width="100%" />
Animated

``` r
anim <- emissions_test %>%
  mutate("Predicted CO2" = predictions) %>%
  filter(CO2 > 50000) %>%
  group_by(Area) %>%
  filter(n() > 10) %>%
  ungroup() %>%
  pivot_longer(
    cols = c("CO2", "Predicted CO2"),
    names_to = "metric",
    values_to = "value"
  ) %>%
  ggplot(aes(x = Year, y = value, group = Area, color = Area)) +
  geom_line() +
  geom_point() +
  theme(legend.position = "none") +
  facet_wrap(~ metric) +
  transition_reveal(Year)
animate(anim, renderer = ffmpeg_renderer(format = "webm"), width=800)
```

<!--html_preserve-->

<video controls autoplay width="100%">

<source src="data:video/webm;base64,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" type="video/mp4">

</video>

<!--/html_preserve-->

## Conclusion

To conclude, it seems that we can indeed predict agricultural emissions
given the raw agricultural and livestock production of the country that
year. Although we used country and year to link the different datasets,
that was for the sole purpose of linking the datasets; our model works
independent of country and year, with raw agricultural and livestock
production as the only necessary inputs to accurately predict
agricultural emissions.

We believe our is a strong predictor as we have conducted multiple
validation metrics, all of which the model has performed well on. Our
first was an evaluation on our held-out test set, with a low r-squared
value of `0.0027`. We also conducted 10-fold cross-validation on our
training set, creating out train / test split 10 ways, and taking a
confidence interval of our r-squared calculated from each sample. This
produced a 95% confidence interval for r-squared of `0.0024 - 0.0046`,
which again indicates a strong model while accounting for uncertainty.
Our final two metrics were more qualitative; we plotted our predictions
to see how they comapred to the actual data. In our first graph, we
found that the predicted CO2 and actual CO2 tracked, following the line
of `y = x`. For our second graph, we presented the same data from the
first graph but in a more visually intuitive manner. We produced two
graphs of predicted and actual CO2, plotting each value over time with
each country as a line. Since our model operated only on raw data of
agricultural and livestock production without the year or country
identified, the fact that we see similar trends in countries over time
further validates our model.
