library(dplyr)
library(ggplot2)


data_cor = read.table(header=TRUE, file="pearsonr.csv")
ggplot(data_cor, aes(x=Representation, y=r, group=1)) + geom_point(size=3) + geom_line(alpha=0.5, size=2) +
  theme(text=element_text(size=22), aspect.ratio=1/3) + ylab("Pearson's r")
ggsave(filename = "../figures/correlation_mfcc.pdf", width=10, height=4)

