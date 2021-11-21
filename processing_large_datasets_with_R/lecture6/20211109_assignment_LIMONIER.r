library(shiny)

# Without reactivity
ui = fluidPage(
  sliderInput(inputId = "num", value = 5, 
              label = "Choose a number", min = 0, max = 100, 
              step=5, animate=TRUE),
  plotOutput("hist"),
  verbatimTextOutput("ttest")
)
server = function(input, output){
  set.seed(1)
  data1 <- rnorm(1000 , 120 , 25)     
  data2 <- rnorm(1000 , 200 , 25) 
  output$hist = renderPlot({
    #Create data
    col1 <- rgb(1,0,0,0.5)
    col2 <- rgb(0,1,0,0.3)
    hist(data1, breaks=20, xlim=c(0,300), col=col1)
    hist(data2, breaks=20, xlim=c(0,300), col=col2, add=T)
    # Add legend
    legend("topleft", legend=c("Data 1","Data 2"), col=c(col1, col2), pt.cex=1, pch=16 )
    })
  output$ttest = renderPrint({t.test(data1, data2)})
}
shinyApp(server = server, ui = ui)





# With reactivity
ui = fluidPage(
  sliderInput(inputId = "num", value = 5, 
              label = "Choose a number", min = 0, max = 100, 
              step=5, animate=TRUE),
  plotOutput("hist"),
  verbatimTextOutput("ttest")
)
server = function(input, output){
  set.seed(1)
  col1 <- rgb(1,0,0,0.5)
  col2 <- rgb(0,1,0,0.3)
  # Create data
  data1 = reactive(rnorm(input$num, 120 , 25))
  data2 = reactive(rnorm(input$num, 200 , 25)) 
  output$hist = renderPlot({
    hist(data1(), breaks=20, xlim=c(0,300), col=col1)
    hist(data2(), breaks=20, xlim=c(0,300), col=col2, add=T)
    # Add legend
    legend("topleft", legend=c("Data 1","Data 2"), col=c(col1, col2), pt.cex=1, pch=16 )
    })
  output$ttest = renderPrint({t.test(data1, data2)})
}
shinyApp(server = server, ui = ui)


