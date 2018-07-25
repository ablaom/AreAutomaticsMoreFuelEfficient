## LOAD LIBRARIES AND GET DATA

using Koala
using KoalaTransforms
import KoalaTransforms.normality
using DataFrames
using CSV
using StatsBase
using HypothesisTests
using Plots
using StatPlots
pyplot() # use python plotting backend

cars = CSV.read("data/cars_data.csv");


## INITIAL LOOK AT DATA

# dump irrelevant features:
cars = cars[[:am, :mpg]];
head(cars)

# check for missing data:
showcols(cars)

# get sample size:
size(cars, 1)

# split our mpg data into manual and automatic:
manual = cars[cars[:am] .== 0,:mpg];
auto = cars[cars[:am] .== 1,:mpg];

# individual sample sizes:
n_manual = length(manual)
n_auto = length(auto)

# gitter line plots of the data:
plt1=plot(; title = "Fuel efficiency for 32 models of auto",
         ylim=(-0.5, 0.5), yscale=:none,
         xlab = "M.P.G.");
scatter!(manual, 0.01*randn(length(manual)), label="manual", ms=5.0);
scatter!(auto, 0.01*randn(length(auto)), label="auto", ms=5.0)
savefig("figures/plt1.png")

# box plot comparison:
plt2=plot(; title = "Box plot comparison of fuel efficiency - auto vs manual",
         xscale=:none, ylabel="M.P.G.");
boxplot!(manual, label="manual");
boxplot!(auto, label="auto")
savefig("figures/plt2.png")

# get bootstrap histograms for the median in each case:
plt3 = plot(; xlab="M. P. G.",
           title="Bootstrap histograms of the median M.P.G.")
bootstrap_histogram_of_median!(manual, label="manual");
bootstrap_histogram_of_median!(auto, label="automatic";)
savefig("figures/plt3.png")

## BOOTSTRAP CONFIDENCE INTERVALS

# do bootstrap simulation of difference of medians:
α = 0.95 # for 95% conf int
n_simulations = 100000
point_estimate = median(auto) - median(manual)
simulated_differences = Float64[]
for i in 1:n_simulations
    mpg_manual = sample(manual, n_manual, replace=true)
    mpg_auto = sample(auto, n_auto, replace=true)
    append!(simulated_differences,  median(mpg_auto) - median(mpg_manual))
end

# calculat pivotal conf int:
left_pvt =  2*point_estimate - quantile(simulated_differences, 1 - α/2)
right_pvt = 2*point_estimate - quantile(simulated_differences, α/2)

# calculate percentile conf int:
left_per = quantile(simulated_differences, α/2)
right_per = quantile(simulated_differences, 1 - α/2)


## T-TEST

# get the box-cox transformation most "normalizing" the data in a
# bigger external set (about 400 data points). The data is from here:
# https://github.com/RodolfoViana/exploratory-data-analysis-dataset-cars
multi = CSV.read("external/cars_multi.csv", rows_for_type_detect=1000,
                 categorical=false, weakrefstrings=false)
mpg_big = multi[:mpg]
length(mpg_big)

# how normal is the data to begin with?
plt4 = histogram(mpg_big, bins=15)
savefig("figures/plt4.png")
normality(mpg_big)

# does log transform improve normality?
normality(log.(mpg_big))

# define a Box-Cox transformer with optimal parameters:
boxcox = UnivariateBoxCoxTransformer()
boxcoxM = Machine(boxcox, mpg_big)
boxcoxM.scheme # the Box-Cox parameters

# transform and retest for normality:
mpg_big2 = transform(boxcoxM, mpg_big);
normality(mpg_big2)

# transform the supplied data sets:
manual2 = transform(boxcoxM, manual);
auto2 = transform(boxcoxM, auto);

# compute p-value for two-sample Welch's t-test:
test = UnequalVarianceTTest(manual2, auto2)
pvalue(test)





