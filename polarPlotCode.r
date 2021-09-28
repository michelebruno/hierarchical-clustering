stars <- read.csv("./hygdata_v3_CLEAN_1K_v1.csv")
library(plotly)

fig <- plot_ly(
    type = 'scatterpolar',

    mode = 'markers',
    
)
fig <- fig %>%
    add_trace(
        r = c <- (stars$dist),
        theta = c <- (stars$deg),
        marker = list(
            color = '#FFF',
            size = 4,
            opacity = c <- (stars$opacity^2),
            line = list(
                width = 0
            )
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
            tickwidth = 1,
            linewidth = 1,
            opacity = 0.1,
            layer = 'below traces'
        ),
        radialaxis = list(
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