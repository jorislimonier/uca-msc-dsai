library(dplyr)
library(ggplot2)
library(shiny)

# Exercise 1
# With reactivity
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
    reactive_filter <- reactive(
        ds %>%
        filter(Price >= input$price[1], 
            Price <= input$price[2], 
            Type == input$type, 
            Country == input$country
        )
    )
    output$hist = renderPlot({
            ggplot(reactive_filter(), aes(x=Alcohol_Content)) + geom_histogram()
    })
    output$summary = renderPrint({
        summary(reactive_filter()$Alcohol_Content)
    })
    output$table = renderDataTable({
        reactive_filter()
    })
    output$text_info = renderText({
        paste("This dataset has ", nrow(reactive_filter()), " rows and ", ncol(reactive_filter()), " columns")
    })
}

shinyApp(server=server, ui=ui)