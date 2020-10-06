# ST558-RProj2 
## Mana Azizsoltani  
## Fall 2020  
This repository is for the second project of ST558 where we are to use two tree-based machine learning methods to try and predict the number of shares for a given internet news article.  

# Links to Sub-Documents  

  * [Monday](mondayAnalysis.md)  
  * [Tuesday](tuesdayAnalysis.md)  
  * [Wednesday](wednesdayAnalysis.md)  
  * [Thursday](thursdayAnalysis.md)  
  * [Friday](fridaydayAnalysis.md)  
  * [Saturday](saturdayAnalysis.md)  
  * [Sunday](sundayAnalysis.md)    

# Required Packages  

  * `knitr`  
  * `gridExtra`  
  * `rmarkdown`  
  * `caret`  
  * `tidyverse`  
  * `class`  
  * `randomForest`  
  * `gbm`  
  * `klaR`  

# Render Function  
```
dayofweek <- c("monday", "tuesday", "wednesday", "thursday",
               "friday", "saturday", "sunday")
output_file <- paste0(dayofweek, "Analysis", ".md")
params = lapply(dayofweek, FUN = function(x){list(day = x)})
reports <- tibble(output_file, params)
apply(reports, MARGIN = 1, FUN = function(x){
                             render(input = "AzizProj2.Rmd", output_file = x[[1]], params = x[[2]])
                             })
```
