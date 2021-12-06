setwd("/home/joris/Documents/uca-msc-dsai/R_processing_large_datasets/exam")
source("./ex1_movie.r", local = TRUE)

# Exercise 6
library(shiny)
library(DT)

ui <- fluidPage(
        headerPanel("Exercise 1"),
        sidebarLayout(
                sidebarPanel(
                        h2("Question 1"),
                        br(),
                        h3("Columns"),
                        textOutput("colnames"),
                        h3("Dimension of the data:"),
                        textOutput("dimensions"),
                ),
                mainPanel(
                        h3("Import and show the data"),
                        DT::dataTableOutput("dataset"),
                )
        ),
        sidebarLayout(
                sidebarPanel(
                        h2("Question 2"),
                        br(),
                        h4("Correlation between tickets sold and sales"),
                        textOutput("corr_scatter"),
                        p("This is expected since there is a
                        direct relationship between the number
                         of tickets sold and the money gained."),
                ),
                mainPanel(
                        plotOutput("scat_ticket_gross"),
                )
        ),
        sidebarLayout(
                sidebarPanel(
                        h2("Question 3"),
                ),
                mainPanel(
                        plotOutput("boxplot_hist"),
                )
        ),
        sidebarLayout(
                sidebarPanel(
                        sliderInput(
                                "nb_bins", "Number of bins",
                                min = 1,
                                max = 50,
                                value = 25,
                                width = "100%",
                                animate = animationOptions(
                                        interval = 500,
                                        loop = FALSE,
                                        playButton = NULL,
                                        pauseButton = NULL
                                )
                        ),
                ),
                mainPanel(
                        plotOutput("hist_ticket_sales"),
                )
        ),
        sidebarLayout(
                sidebarPanel(
                        h2("Question 4"),
                        checkboxInput(
                                inputId = "by_genre",
                                label = "By genre",
                                value = TRUE
                        ),
                        checkboxInput(
                                inputId = "by_distr",
                                label = "By distributor",
                                value = TRUE
                        ),
                        selectInput(
                                inputId = "metric",
                                label = "Metric",
                                choices = c("ticket", "sales")
                        ),
                ),
                mainPanel(
                        DT::dataTableOutput("av_metric"),
                )
        ),
)
server <- function(input, output) {
        output$dataset <- DT::renderDataTable({
                movies
        })
        output$colnames <- renderText(paste(colnames(movies)))
        output$dimensions <- renderText(
                paste(nrow(movies), "rows and ", ncol(movies), "columns")
        )
        output$scat_ticket_gross <- renderPlot({
                scat_ticket_gross
        })
        output$corr_scatter <- renderText(paste(corr_scatter))
        output$boxplot_hist <- renderPlot({
                grid.arrange(
                        boxplot_all,
                        hist_genre,
                        ncol = 2,
                        widths = c(1.8, 2.2)
                )
        })

        nb_bins_react <- reactive({
                input$nb_bins
        })
        output$hist_ticket_sales <- renderPlot({
                hist_ticket_sales(nb_bins_react())
        })

        by_genre_react <- reactive({
                input$by_genre
        })
        by_distr_react <- reactive({
                input$by_distr
        })
        metric_react <- reactive({
                input$metric
        })

        decide_render_av_metric <- function(by_genre_local,
                                            by_distr_local,
                                            metric_local) {
                if (as.numeric(by_genre_local) == FALSE & as.numeric(by_distr_local) == FALSE) {
                        return(data.frame("No value selected" = "Select at least one value to group by"))
                } else {
                        return(
                                av_metric(
                                        by_genre = by_genre_local,
                                        by_distr = by_distr_local,
                                        metric = metric_local
                                )
                        )
                }
        }
        output$av_metric <- DT::renderDataTable({
                decide_render_av_metric(
                        by_genre_local = by_genre_react(),
                        by_distr_local = by_distr_react(),
                        metric_local = metric_react()
                )
        })
}

shiny::runApp(shinyApp(ui = ui, server = server), port = 5000)