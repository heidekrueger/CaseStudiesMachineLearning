values_100 = as.matrix(read.table("100_10.txt", sep=","), ncol = 5+29)
values_500 = as.matrix(read.table("500_10.txt", sep=","), ncol = 5+29)
values_1000 = as.matrix(read.table("1000_10.txt", sep=","), ncol = 5+29)
values_10000 = as.matrix(read.table("10000_10.txt", sep=","), ncol = 5+29)

plot_column <- function(name, col_x, col_y) {
    jpeg(name)
    max_x = max( c(values_100[, col_x], values_500[, col_x], values_1000[, col_x], values_10000[, col_x]))
    min_x = min( c(values_100[, col_x], values_500[, col_x], values_1000[, col_x], values_10000[, col_x]))
    max_y = max( c(values_100[, col_y], values_500[, col_y], values_1000[, col_y], values_10000[, col_y]))
    min_y = min( c(values_100[, col_y], values_500[, col_y], values_1000[, col_y], values_10000[, col_y]))
    
    f_100 = values_100[, col_y]
    f_500 = values_500[, col_y]
    f_1000 = values_1000[, col_y]
    f_10000 = values_10000[, col_y]
    plot(c(min_x, max_x), c(min_y, max_y))
    lines(values_100[, col_x], f_100, col="black")
    lines(values_500[, col_x], f_500, col='blue')
    lines(values_1000[, col_x], f_1000, col='green')
    lines(values_10000[, col_x], f_10000, col='red')
    dev.off()
}

plot_column("plots/adp_f.jpg", 3, 4)
plot_column("plots/fevals_f.jpg", 1, 4)
plot_column("plots/gevals_f.jpg", 2, 4)
plot_column("plots/adp_g.jpg", 3, 5)
plot_column("plots/fevals_g.jpg", 1, 5)
plot_column("plots/adp_w1.jpg", 3, 6)
plot_column("plots/adp_w2.jpg", 3, 7)
plot_column("plots/adp_w3.jpg", 3, 8)