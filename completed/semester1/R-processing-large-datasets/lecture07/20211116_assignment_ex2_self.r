library(dplyr)
library(ggplot2)
library(gridExtra)
library(shiny)
library(scales)

# Exercise 2
make_sim <- function(kind) {
    # set.seed(42)
    nb_val <- 100
    x_val <- sample(seq(-10, 10, .1), nb_val)
    noise <- rnorm(nb_val, 0, 1.5)
    if (kind == "linear_up"){
        y_val <- (2*x_val + 1) + noise
    }
    else if (kind == "linear_down"){
        y_val <- (-2*x_val + 1) + noise
    }
    else if (kind == "curved_up"){
        y_val <- (2*x_val^2 + 1) + noise
    }
    else if (kind == "curved_down"){
        y_val <- (-2*x_val^2 + 1) + noise
    }
    else {
        return("problem")
    }
    return(data.frame(x=x_val, y=y_val))
}

ui = fluidPage(
    titlePanel("Diagnostics for simple linear regression"),
    sidebarPanel(
        radioButtons("type", "Select a trend",
        c("Linear up"="linear_up", 
        "Linear down"="linear_down", 
        "Curved up"="curved_up", 
        "Curved down"="curved_down")
        ),
        checkboxInput("show_residuals", "Show residuals")
    ),
    mainPanel(
        plotOutput("scatter"),
        br(),
        plotOutput("residuals")
)
)
server = function(input, output){
    sim <- reactive({make_sim(input$type)}) 
    output$scatter = renderPlot({
        x_data <- sim()$x
        y_data <- sim()$y
        df_sim <- data.frame(x_data,y_data)
        if (grepl("linear", input$type)){
            p1 <- ggplot(df_sim, aes(x_data,y_data)) +
            geom_point() +
            geom_smooth(formula=y_data~x_data, method="lm", se=FALSE)
        } else if (grepl("curved", input$type)){
            p1 <- ggplot(df_sim, aes(x_data,y_data)) +
            geom_point() +
            geom_smooth(formula=y_data~x_data, se=FALSE)
        }
        p1
    })
    output$residuals = renderPlot({
        x_data <- sim()$x
        y_data <- sim()$y
        df_sim <- data.frame(x_data,y_data)
        mod <- lm(y_data~x_data)

        p2 <- ggplot(mod, aes(x=.fitted, y=.resid)) +
        geom_point()
        p3 <-ggplot(mod, aes(.resid)) +
        geom_histogram()
        # +
        # ggplot(mod, aes((.resid)) +
        # geom_density()

        grid.arrange(p2, p3, ncol=3)
    })
}

shinyApp(server=server, ui=ui)


