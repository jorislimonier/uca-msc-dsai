source("./ex1_movie.r", local=TRUE)

# Exercise 6
library(shiny)
library(DT)

ui <- fluidPage(
        titlePanel("Exercise 1"),
        h2("Question 1"),
        br(),
        sidebarPanel(
                h4("Columns:"),
                textOutput("colnames"),
                h4("Dimension of the data:"),
                textOutput("dimensions"),
        ),

        mainPanel(
                h3("Import and show the data"),
                DT::dataTableOutput("dataset"),
                
                h2("Question 2"),
                plotOutput("scat_ticket_gross"),
                h3("Correlation between tickets sold and sales"),
                textOutput("corr_scatter"),
                p("This is expected since there is a direct relationship 
                between the number of tickets sold and the money gained."),
                
                h2("Question 3"),
                plotOutput("boxplot_hist"),
                sliderInput("nb_bins", "Number of bins", min=1, max=50, value=25, width="100%", animate=TRUE),
                plotOutput("hist_ticket_sales"),
                
                h2("Question 4"),
        )
)
server <- function(input, output){
        output$dataset = DT::renderDataTable({movies})
        output$colnames = renderText(paste(colnames(movies)))
        output$dimensions = renderText(
                paste(nrow(movies), "rows and ", ncol(movies), "columns")
        )
        output$scat_ticket_gross = renderPlot({scat_ticket_gross})
        output$corr_scatter = renderText(paste(corr_scatter))
        output$boxplot_hist = renderPlot({
                grid.arrange(boxplot_all, hist_genre, ncol=2, widths=c(1.8,2.2))
        })

        nb_bins_react <- reactive({input$nb_bins})
        output$hist_ticket_sales = renderPlot({
                hist_ticket_sales(nb_bins_react())
        })
}
shiny::runApp(shinyApp(ui=ui, server=server), port=5000)
