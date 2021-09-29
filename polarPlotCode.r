stars <- read.csv("./database_NorthernEmisphere.csv")
library(plotly)

fig <- plot_ly(
    type = 'scatterpolar',
    mode = 'markers',
)

fig <- fig %>%
    add_trace(
        name = c <- (stars$proper),
        r = c <- (stars$dec),
        theta = c <- ((stars$ra/24)*360),
        marker = list(
            color = '#FFF',
            size = 2,
            opacity = 1
        ),
        showlegend = F
    ) 
fig <- layout(
    fig,
    ##title = "Stars from HYG database (v3.0)",
    font = list(
        family = 'Arial',
        size = 12,
        color = '#FFF'
    ),
    showlegend = F,
    paper_bgcolor = "rgb(0, 0, 0)",
    polar = list(
        bgcolor = "rgb(0, 0, 0)",
        angularaxis = list(
            range = c(0, 24),
            tickwidth = 1,
            linewidth = 1,
            opacity = 0.1,
            layer = 'below traces'
        ),
        radialaxis = list(
            range = c(90, 0),
            side = 'counterclockwise',
            showline = F,
            linewidth = 1,
            tickwidth = 1,
            gridcolor = '#333',
            ##opacity = 0.1,
            gridwidth = 0.1
        )
    )
)

fig