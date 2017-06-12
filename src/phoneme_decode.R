library(ggplot2)
library(dplyr)
library(reshape2)
repr.names <- c("MFCC", "Conv", "Rec1", "Rec2", "Rec3", "Rec4", "Rec5")
data <- read.table("err_boot.csv",
                  col.names = repr.names)

ggplot(melt(data), aes(x=variable, y=value)) + geom_boxplot(notch = TRUE) +
  xlab("Representation") +
  ylab("Error rate") +
  theme(aspect.ratio=3/4, text=element_text(size=22))

ggsave("../figures/decode.pdf", width=10, height=8)
