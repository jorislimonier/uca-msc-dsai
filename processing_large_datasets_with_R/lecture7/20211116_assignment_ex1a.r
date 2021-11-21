library(dplyr)
library(ggplot2)
library(shiny)

# Exercise 1
# Without reactivity
ds <- read.csv("lecture7/bcl-data.csv")

ui = fluidPage(
    titlePanel(
        "BC Liquor Store prices"
    ),
    sidebarPanel(
        sliderInput(
            inputId = "price", value = c(25,40), label = "Price", min = 0, max = 100, step=5, pre="$"), 
            radioButtons("type", "Product type", 
            c("Beer"="BEER", 
                "Refreshment" = "REFRESHMENT", 
                "Spirits"="SPIRITS", 
                "Wine"="WINE")),
        selectInput(
            "country", "Country", 
            c("Canada"="CANADA", 
                "France"="FRANCE",
                "Italy"="ITALY",
                "USA"="UNITED STATES OF AMERICA")
        ),
        h4(textOutput("text_info"))
    ),
    mainPanel(
        plotOutput("hist"),
        verbatimTextOutput("summary"),
        dataTableOutput("table")
    )
)

server = function(input, output){
    output$hist = renderPlot({
            filter_ds <- ds %>%
                filter(Price >= input$price[1], 
                    Price <= input$price[2], 
                    Type == input$type, 
                    Country == input$country
                )
            ggplot(filter_ds, aes(x=Alcohol_Content)) + geom_histogram()
    })
    output$summary = renderPrint({
        filter_ds <- ds %>%
                filter(Price >= input$price[1], 
                    Price <= input$price[2], 
                    Type == input$type, 
                    Country == input$country
                )
        summary(filter_ds$Alcohol_Content)
    })
    output$table = renderDataTable({
        filter_ds <- ds %>%
                filter(Price >= input$price[1], 
                    Price <= input$price[2], 
                    Type == input$type, 
                    Country == input$country
                )
        filter_ds
    })
    output$text_info = renderText({
        filter_ds <- ds %>%
                filter(Price >= input$price[1], 
                    Price <= input$price[2], 
                    Type == input$type, 
                    Country == input$country
                )
        paste("This dataset has ", nrow(filter_ds), " rows and ", ncol(filter_ds), " columns")
    })
}

shinyApp(server=server, ui=ui)